from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from bot_graph import workflow
from langchain_core.messages import HumanMessage
import logging
import tempfile
from utils import parse_file, extract_data_from_resume , analyze_resume
from database import insert_extracted_data , get_thread_data , test_connection , drop_table , create_table
from typing import List

# ====================== Logging Setup ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RecruitmentAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = "static"
os.makedirs(STATIC, exist_ok=True)


# ====================== Homepage ======================
@app.get("/")
def homepage():
    try:
        file_path = os.path.join(STATIC, "chatbot_UI.html")
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Homepage UI file not found."
            )
        return FileResponse(file_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve homepage: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while loading homepage."
        )



# ====================== Upload Endpoint (Temp File → Parse → Delete) ======================
@app.post("/upload")
async def upload_files(
    resume_files: List[UploadFile] = File(..., description="Multiple resume files"),          
    job_description_file: UploadFile = File(..., description="Single job description file"),
    thread_id: str = Form(...),
):
    if not thread_id or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id is required and cannot be empty."
        )

    if not resume_files or not any(f.filename for f in resume_files):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one resume file is required."
        )

    if not job_description_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description file is required."
        )

    # ------------------------------------------------------------------
    # 1. Initialise tracking variables
    # ------------------------------------------------------------------
    success = {"resumes": [], "job_description": False}
    temp_resume_paths: List[str] = []
    temp_jd_path: str | None = None

    resumes = []  # To store parsed resume texts
    job_description_data: str | None = None

    try:
        # ------------------------------------------------------------------
        # 2. PROCESS MULTIPLE RESUMES
        # ------------------------------------------------------------------
        for idx, resume_file in enumerate(resume_files):
            if not resume_file.filename:
                continue  # Skip empty entries

            ext = resume_file.filename.rsplit(".", 1)[-1].lower()
            if ext not in {"pdf", "txt", "docx"}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported resume file type: .{ext} in file '{resume_file.filename}'"
                )

            raw_bytes = await resume_file.read()
            if not raw_bytes:
                logger.warning(f"[THREAD {thread_id}] Resume {idx} is empty: {resume_file.filename}")
                continue

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            temp_file.write(raw_bytes)
            temp_file.close()
            temp_path = temp_file.name
            temp_resume_paths.append(temp_path)

            logger.info(f"[THREAD {thread_id}] Resume {idx} temp file: {temp_path}")

            try:
                parsed_text = parse_file(temp_path)
                resumes.append(parsed_text)
                success["resumes"].append({
                    "filename": resume_file.filename,
                    "parsed": True,
                    "length": len(parsed_text)
                })
                logger.info(f"[THREAD {thread_id}] Resume {idx} parsed: {resume_file.filename} ({len(parsed_text)} chars)")
            except Exception as e:
                success["resumes"].append({
                    "filename": resume_file.filename,
                    "parsed": False,
                    "error": str(e)
                })
                logger.error(f"[THREAD {thread_id}] Failed to parse resume {idx}: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    temp_resume_paths.remove(temp_path)
                    logger.info(f"[THREAD {thread_id}] Deleted resume temp file: {temp_path}")

        if not resumes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No resumes were successfully parsed."
            )

        # ------------------------------------------------------------------
        # 3. JOB DESCRIPTION (Single File)
        # ------------------------------------------------------------------
        ext = job_description_file.filename.rsplit(".", 1)[-1].lower()
        if ext not in {"pdf", "txt", "docx"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported JD file type: .{ext}"
            )

        raw_bytes = await job_description_file.read()
        if not raw_bytes:
            raise ValueError("Job description file is empty.")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        temp_file.write(raw_bytes)
        temp_file.close()
        temp_jd_path = temp_file.name

        logger.info(f"[THREAD {thread_id}] JD temp file created: {temp_jd_path}")

        job_description_data = parse_file(temp_jd_path)
        success["job_description"] = True
        logger.info(f"[THREAD {thread_id}] JD parsed successfully (len={len(job_description_data)} chars)")

        # Cleanup JD temp file
        if os.path.exists(temp_jd_path):
            os.unlink(temp_jd_path)
            logger.info(f"[THREAD {thread_id}] Deleted JD temp file: {temp_jd_path}")
        temp_jd_path = None

        # ------------------------------------------------------------------
        # 4. ANALYSIS & EXTRACTION (Now over multiple resumes)
        # ------------------------------------------------------------------
        try:
            logger.info(f"Starting analysis for {len(resumes)} resume(s) in thread {thread_id}")

            # Combine all resume texts for analysis (or process individually as needed)
            combined_resume_text = "\n\n--- RESUME SEPARATOR ---\n\n".join(resumes)

            # Extract data from **first** resume only (you can modify logic if needed)
            # Or loop over all and store multiple entries
            first_resume_data = resumes[0]
            extracted_resume_data = extract_data_from_resume(resume_data=first_resume_data)

            # Analyze all resumes against JD
            analysis = analyze_resume(
                resume_data=combined_resume_text,  # Pass all resumes
                job_description=job_description_data
            )
            logger.info(f"Analysis completed for thread {thread_id}")

            # Insert into DB (modify schema if storing multiple resumes)
            email_address = extracted_resume_data.get("email_address", "")
            linkedin_url = extracted_resume_data.get("linkedin_url", "")
            total_experience = int(extracted_resume_data.get("total_experience", 0))
            skills = extracted_resume_data.get("skills", [])
            education = extracted_resume_data.get("education", "")
            work_experience = extracted_resume_data.get("work_experience", "")
            projects = extracted_resume_data.get("projects", "")

            insert_extracted_data(
                thread_id=thread_id,
                email_address=email_address,
                linkedin_url=linkedin_url,
                total_experience=total_experience,
                skills=skills,
                education=education,
                work_experience=work_experience,
                projects=projects,
                analysis=analysis,
                resume_data=combined_resume_text,  # Store all resumes
                job_description_data=job_description_data
            )

            logger.info("Extracted data and analysis saved to DB")

        except Exception as e:
            logger.error(f"Analysis failed for thread {thread_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Resume analysis failed: {str(e)}"
            )

        return JSONResponse({
            "success": True,
            "uploaded": success,
            "both_uploaded": len(resumes) > 0 and success["job_description"],
            "resume_count": len(resumes)
        })

    except HTTPException:
        # Cleanup any remaining temp files
        for path in [temp_jd_path] + temp_resume_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Cleaned up temp file on error: {path}")
                except:
                    pass
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in /upload for thread {thread_id}: {str(e)}")
        for path in [temp_jd_path] + temp_resume_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        return JSONResponse({
            "success": False,
            "uploaded": success,
            "both_uploaded": False,
            "analysis": None,
            "error": "An unexpected error occurred during upload."
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post('/query')
def answer_query(
    thread_id : str = Form(...),
    query : str = Form(...)
):
    
    # check if thread id is present or not in database
    thread_id_data = get_thread_data(thread_id=thread_id)


    logger.info(f"Thread data \n :- {thread_id_data}")

    if thread_id_data:
        # If thread exists then get the resume data and job description
        analyzed_resume_data = thread_id[-3] 
        resume_data = thread_id_data[-2]
        job_description_data = thread_id_data[-1]
        
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "messages": [HumanMessage(content=query)],
            "conversation_thread": thread_id,
            "analyzed_resume_data": analyzed_resume_data,
            "job_description":job_description_data,
            "resume_data": resume_data
        }

        try:
            result = workflow.invoke(input_state, config=config)
            ai_message = result["messages"][-1]
            response_content = ai_message.content or "No response generated."
            return JSONResponse({"response": response_content})
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

        
    else:
         raise HTTPException(status_code=400, detail="thread_id is required.")
    



# ====================== Health Check ======================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# ====================== Run Server ======================
if __name__ == "__main__":
    test_connection()
    drop_table()
    create_table()
    port = 3333
    logger.info(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)