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
from database_sqlite import insert_extracted_data , get_thread_data , test_connection , drop_table , create_table
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
    temp_jd_path: str | None = None
    job_description_data: str | None = None
    processed_count = 0
    failed_analyses = []


    logger.info("Upload started ... ")
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
        # 3. PROCESS EACH RESUME: PARSE → EXTRACT → ANALYZE → STORE → CLEANUP
        # ------------------------------------------------------------------
        for idx, resume_file in enumerate(resume_files):
            if not resume_file.filename:
                continue  # Skip empty entries

            temp_resume_path: str | None = None

            try:
                # Validate file type
                ext = resume_file.filename.rsplit(".", 1)[-1].lower()
                if ext not in {"pdf", "txt", "docx"}:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported resume file type: .{ext} in file '{resume_file.filename}'"
                    )

                # Read and save to temp file
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

                # Parse the resume
                parsed_text = parse_file(temp_resume_path)
                logger.info(f"[THREAD {thread_id}] Resume {idx} parsed: {resume_file.filename} ({len(parsed_text)} chars)")

                # Extract data from resume
                extracted_resume_data = extract_data_from_resume(resume_data=parsed_text)
                logger.info(f"[THREAD {thread_id}] Resume {idx} data extracted")

                

                # Extract fields for DB insertion
                candidate_name = extracted_resume_data.get("candidate_name","")
                email_address = extracted_resume_data.get("email_address", "")
                linkedin_url = extracted_resume_data.get("linkedin_url", "")
                total_experience = int(extracted_resume_data.get("total_experience", 0))
                skills = extracted_resume_data.get("skills", [])
                education = extracted_resume_data.get("education", "")
                work_experience = extracted_resume_data.get("work_experience", "")
                projects = extracted_resume_data.get("projects", "")


                # Analyze resume against job description
                analysis_result = analyze_resume(
                    extracted_resume_data=extracted_resume_data,
                    job_description=job_description_data
                )

                fit_score = analysis_result['fit_score']
                analysis_summary = analysis_result['analysis_summary']


                logger.info(f"[THREAD {thread_id}] Resume {idx} analysis completed")



                # Insert into database
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
                    job_description = job_description_data,
                    fit_score=fit_score,
                    analysis=analysis_summary
                )

                processed_count += 1
                success["resumes"].append({
                    "filename": resume_file.filename,
                    "parsed": True,
                    "length": len(parsed_text)
                })
                logger.info(f"[THREAD {thread_id}] Resume {idx} ({resume_file.filename}) saved to DB")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[THREAD {thread_id}] Failed to process resume {idx} ({resume_file.filename}): {error_msg}")
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
                # Cleanup temp file immediately after processing
                if temp_resume_path and os.path.exists(temp_resume_path):
                    os.unlink(temp_resume_path)
                    logger.info(f"[THREAD {thread_id}] Deleted resume temp file: {temp_resume_path}")

        # Check if at least one resume was processed successfully
        if processed_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No resumes were successfully processed."
            )

        logger.info(f"Successfully processed {processed_count}/{len(resume_files)} resumes for thread {thread_id}")

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
        # Cleanup JD temp file if still exists
        if temp_jd_path and os.path.exists(temp_jd_path):
            try:
                os.unlink(temp_jd_path)
                logger.info(f"Cleaned up JD temp file on error: {temp_jd_path}")
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


@app.post('/query')
def answer_query(
    thread_id : str = Form(...),
    query : str = Form(...)
):
    try:
        # check if thread id is present or not in database
        thread_id_data = get_thread_data(thread_id=thread_id)

        
        if thread_id_data:
            # If thread exists then get the analysis , fit score and job description.
            job_description = thread_id_data[0][-3]


            all_analysis_data = """"""

            for data in thread_id_data:
                fitscore = data[-2]
                analysis = data[-1]

                all_analysis_data += "\n" + f"""
                ANALYSIS :- {analysis}

                FITSCORE :- {fitscore}
                """ + "\n"
            
            config = {"configurable": {"thread_id": thread_id}}
            input_state = {
                "messages": [HumanMessage(content=query)],
                "analyzed_resume_data": all_analysis_data,
                "job_description":job_description,
            }

            
            result = workflow.invoke(input_state, config=config)
            ai_message = result["messages"][-1]
            response_content = ai_message.content or "No response generated."
            return JSONResponse({"response": response_content})
            
            
        else:
            raise HTTPException(status_code=400, detail="thread_id is required.")
    except Exception as e :
        logger.error(f"Query failed: {str(e)}")



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
    
    uvicorn.run(
    "server:app",
    host="0.0.0.0",
    port=port,
    reload=False,
    log_level="info"  # This enables Uvicorn logs
)