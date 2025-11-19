import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import json
import os
import re

# --- CONFIGURATION (Load from Environment Variables) ---
# When deployed, host services use environment variables (secrets)
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    # Use dummy key for local testing if not set
    GEMINI_API_KEY = "AIzaSyDdOkFYez625UjfgfOSefxA4JGdmLg_PXQ" 

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI()

# -----------------------------
# UTILITY FUNCTION: Prepare Multimodal Parts for Gemini
# -----------------------------
def get_gemini_parts(file_bytes: bytes, mime_type: str):
    """Handles file bytes and prepares it as an inline Part object."""
    if file_bytes:
        return [types.Part.from_bytes(data=file_bytes, mime_type=mime_type)]
    return []

# -----------------------------
# GEMINI AI CORE FUNCTIONS (Modified for API I/O)
# Note: Simplified to remove st.session_state dependencies
# -----------------------------

# This function is now the central intelligence for the entire app.
# It will perform keyword expansion, search, AND formatting in one go.
async def run_full_job_search(
    job_role: str, 
    qualification: str, 
    location: str, 
    custom_keywords: str,
    resume_parts: list
):
    """Consolidated logic for keyword expansion, search, and result formatting."""
    
    # 1. Expand Keywords (same logic as before)
    text_prompt = f"""
    You are a job AI assistant. Your task is to extract skills and experiences from the provided RESUME (if available) and combine them with the user's explicit inputs below.
    User Input: Job Role: {job_role}, Qualification: {qualification}, Location: {location}, Custom Keywords: {custom_keywords}
    Analyze the resume for relevant technical skills, past job titles, and industry focus. Generate a comprehensive, comma-separated list of related keywords and roles for the final search query.
    Return ONLY the comma-separated keywords and roles, no explanation.
    """
    
    contents = resume_parts[:]
    contents.append(types.Part.from_text(text=text_prompt))
    
    keyword_response = client.models.generate_content(
        model="gemini-2.5-flash", contents=contents
    )
    
    keywords_list = [k.strip() for k in keyword_response.text.split(",") if k.strip()]
    search_query = " ".join(keywords_list + [qualification, location])
    
    # 2. Consolidated Search and Analysis (CRITICAL)
    # Ask Gemini to use search tool AND analyze results for formatting.
    search_analysis_prompt = f"""
    Based on the refined search query: "{search_query}", use the Google Search tool to find the top 5 most relevant job postings.
    For each posting found, extract the 'title', the 'link', and the original search 'snippet'. 
    Then, for each posting, perform a secondary analysis to extract these three key points into a JSON sub-object named 'analysis':
    1.  Type: (e.g., Full-time, Remote)
    2.  Requirement: (The single most essential skill)
    3.  USP: (Unique Selling Point of the job or company)
    Return the final, structured list of jobs as a strict JSON array.
    """
    
    final_contents = [types.Part.from_text(text=search_analysis_prompt)]
    search_tool = {"google_search": {}}

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_contents,
        config={
            "tools": [search_tool],
            "response_mime_type": "application/json",
            "system_instruction": "You are a specialized job search tool that strictly outputs a JSON array containing job objects. Each job object must include a nested 'analysis' object with Type, Requirement, and USP fields."
        }
    )
    
    # 3. Parse and Clean Results
    text = final_response.text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        
    results = json.loads(text)
    
    # Return the full search response including the generated keywords
    return {"search_query": search_query, "keywords": keywords_list, "results": results}

# -----------------------------
# FASTAPI ENDPOINT
# -----------------------------

@app.post("/search-jobs")
async def search_jobs(
    job_role: str = Form(...),
    qualification: str = Form(...),
    location: str = Form(...),
    custom_keywords: str = Form(""),
    resume_file: UploadFile = File(None),
):
    """Receives user data and returns job search results."""
    
    # Process Resume File
    resume_parts = []
    if resume_file:
        try:
            file_bytes = await resume_file.read()
            mime_type = resume_file.content_type
            resume_parts = get_gemini_parts(file_bytes, mime_type)
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"File reading failed: {e}"})

    try:
        # Run the consolidated search and analysis
        search_data = await run_full_job_search(
            job_role, qualification, location, custom_keywords, resume_parts
        )
        return JSONResponse(content=search_data)
        
    except Exception as e:
        # Log the full exception for debugging
        print(f"Gemini/Search Error: {e}")
        raise HTTPException(
            status_code=500, detail="The AI search service failed to process the request. Try refining your query."
        )

# Boilerplate for running locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
