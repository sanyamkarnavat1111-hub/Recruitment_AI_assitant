from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from bot_graph import workflow
from langchain_core.messages import HumanMessage, AIMessage
import logging
import tempfile
from utils import parse_file, extract_data_from_resume, analyze_resume
from database_sqlite import insert_extracted_data, get_thread_data, test_connection, drop_table, create_table
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


# ====================== Upload Endpoint (Initialize Thread State) ======================
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
    temp_jd_path: str | None = None
    job_description_data: str | None = None
    processed_count = 0
    failed_analyses = []

    logger.info(f"Upload started for thread: {thread_id}")

    try:
        # ------------------------------------------------------------------
        # 2. PROCESS JOB DESCRIPTION FIRST
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
        # 3. PROCESS EACH RESUME
        # ------------------------------------------------------------------
        all_analysis_data = ""

        for idx, resume_file in enumerate(resume_files):
            if not resume_file.filename:
                continue

            temp_resume_path: str | None = None

            try:
                ext = resume_file.filename.rsplit(".", 1)[-1].lower()
                if ext not in {"pdf", "txt", "docx"}:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported resume file type: .{ext} in file '{resume_file.filename}'"
                    )

                raw_bytes = await resume_file.read()
                if not raw_bytes:
                    logger.warning(f"[THREAD {thread_id}] Resume {idx} is empty: {resume_file.filename}")
                    success["resumes"].append({
                        "filename": resume_file.filename,
                        "parsed": False,
                        "error": "Empty file"
                    })
                    continue

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
                temp_file.write(raw_bytes)
                temp_file.close()
                temp_resume_path = temp_file.name

                logger.info(f"[THREAD {thread_id}] Resume {idx} temp file: {temp_resume_path}")

                # Parse resume
                parsed_text = parse_file(temp_resume_path)
                logger.info(f"[THREAD {thread_id}] Resume {idx} parsed: {resume_file.filename}")

                # Extract data
                extracted_resume_data = extract_data_from_resume(resume_data=parsed_text)
                logger.info(f"[THREAD {thread_id}] Resume {idx} data extracted")

                # Extract fields for DB
                candidate_name = extracted_resume_data.get("candidate_name", "")
                email_address = extracted_resume_data.get("email_address", "")
                linkedin_url = extracted_resume_data.get("linkedin_url", "")
                total_experience = int(extracted_resume_data.get("total_experience", 0))
                skills = extracted_resume_data.get("skills", [])
                education = extracted_resume_data.get("education", "")
                work_experience = extracted_resume_data.get("work_experience", "")
                projects = extracted_resume_data.get("projects", "")

                # Analyze resume
                analysis_result = analyze_resume(
                    extracted_resume_data=extracted_resume_data,
                    job_description=job_description_data
                )

                fit_score = analysis_result['fit_score']
                analysis_summary = analysis_result['analysis_summary']

                logger.info(f"[THREAD {thread_id}] Resume {idx} analysis completed")

                # Build analysis string for state
                all_analysis_data += f"""
                ANALYSIS :- {analysis_summary}
                FITSCORE :- {fit_score}
                """ + "\n"

                # Insert into DB
                insert_extracted_data(
                    thread_id=thread_id,
                    candidate_name=candidate_name,
                    email_address=email_address,
                    linkedin_url=linkedin_url,
                    total_experience=total_experience,
                    skills=skills,
                    education=education,
                    work_experience=work_experience,
                    projects=projects,
                    job_description=job_description_data,
                    fit_score=fit_score,
                    analysis=analysis_summary
                )

                processed_count += 1
                success["resumes"].append({
                    "filename": resume_file.filename,
                    "parsed": True,
                    "length": len(parsed_text)
                })
                logger.info(f"[THREAD {thread_id}] Resume {idx} saved to DB")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[THREAD {thread_id}] Failed resume {idx}: {error_msg}")
                success["resumes"].append({
                    "filename": resume_file.filename,
                    "parsed": False,
                    "error": error_msg
                })
                failed_analyses.append({
                    "filename": resume_file.filename,
                    "index": idx,
                    "error": error_msg
                })

            finally:
                if temp_resume_path and os.path.exists(temp_resume_path):
                    os.unlink(temp_resume_path)
                    logger.info(f"[THREAD {thread_id}] Deleted resume temp file")

        # ------------------------------------------------------------------
        # 4. Validate & Initialize LangGraph State ONCE
        # ------------------------------------------------------------------
        if processed_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No resumes were successfully processed."
            )

        config = {"configurable": {"thread_id": thread_id}}

        # Initialize state only once
        initial_state = {
            "messages": [],
            "analyzed_resume_data": all_analysis_data.strip(),
            "job_description": job_description_data,
        }

        # Save to LangGraph memory
        workflow.update_state(
            config=config,
            values=initial_state
        )

        logger.info(f"[THREAD {thread_id}] LangGraph state initialized with {processed_count} resumes")

        return JSONResponse({
            "success": True,
            "uploaded": success,
            "both_uploaded": processed_count > 0 and success["job_description"],
            "resume_count": len(resume_files),
            "processed_count": processed_count,
            "failed_count": len(failed_analyses),
            "failed_analyses": failed_analyses if failed_analyses else None
        })

    except HTTPException:
        if temp_jd_path and os.path.exists(temp_jd_path):
            try:
                os.unlink(temp_jd_path)
            except:
                pass
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in /upload for thread {thread_id}: {str(e)}")
        if temp_jd_path and os.path.exists(temp_jd_path):
            try:
                os.unlink(temp_jd_path)
            except:
                pass
        return JSONResponse({
            "success": False,
            "uploaded": success,
            "both_uploaded": False,
            "error": "An unexpected error occurred during upload."
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ====================== Query Endpoint (Uses Persisted State) ======================
@app.post('/query')
def answer_query(
    thread_id: str = Form(...),
    query: str = Form(...)
):
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Check if thread state exists in LangGraph memory
        saved_state = workflow.checkpointer.get(config)
        if not saved_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found. Please upload resumes and job description first."
            )
        
        # Only send the new user message
        input_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # LangGraph loads previous state (analyzed_resume_data, job_description) automatically
        result = workflow.invoke(input_state, config=config)

        # Get AI response
        ai_message = result["messages"][-1]
        if isinstance(ai_message, AIMessage):
            response_content = ai_message.content
        else:
            response_content = str(ai_message)

        return JSONResponse({"response": response_content or "No response generated."})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed for thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query."
        )


# ====================== Health Check ======================
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ====================== Run Server ======================
if __name__ == "__main__":
    test_connection()
    drop_table()
    create_table()
    port = 3333
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )