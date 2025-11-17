from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status, BackgroundTasks , Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from bot_graph import workflow
from langchain_core.messages import HumanMessage, AIMessage
import logging
import tempfile
from utils import (
    parse_file, 
    extract_data_from_resume, 
    analyze_resume, 
    get_fittest_candidates , 
    remove_extra_space,
    summarize_job_description,
    validate_contact_info
)
from database_sqlite import (
    test_connection, drop_table, create_table,
    insert_extracted_data, insert_job_description, get_job_description
)
from typing import List
from dotenv import load_dotenv
from Screening_AI import ScreenAI
from RAG import VectorStorage
import asyncio
import json
import uuid

load_dotenv()

# ====================== Logging Setup ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RecruitmentAI")


# === Initialize AI ===
AI = ScreenAI()

# === Initialize Vector store for RAG ===
vectorStore = VectorStorage()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency function to check thread_id
def check_thread_id(thread_id: str = Form(...)) -> str:
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="Thread ID is required.")
    return thread_id


# ====================== Homepage ======================
@app.get("/")
def homepage():
    try:
        file_path = os.path.join(os.environ['STATIC_DIR'], "chatbot_UI.html")
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

# ====================== Upload Job Description ======================
@app.post("/upload_job_description")
async def upload_jd(
    thread_id: str = Depends(check_thread_id),
    job_description_file: UploadFile = File(..., description="Single job description file"),
):
    temp_jd_path: str | None = None
    try:

        if not job_description_file.filename:
            raise HTTPException(status_code=400, detail="Job description file is required.")

        ext = job_description_file.filename.rsplit(".", 1)[-1].lower()
        if ext not in {"pdf", "txt", "docx"}:
            raise HTTPException(status_code=400, detail=f"Unsupported JD file type: .{ext}")

        raw_bytes = await job_description_file.read()
        if not raw_bytes:
            raise ValueError("Job description file is empty.")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        temp_file.write(raw_bytes)
        temp_file.close()
        temp_jd_path = temp_file.name

        logger.info(f"[THREAD {thread_id}] JD temp file created: {temp_jd_path}")

        job_description_data = parse_file(temp_jd_path)
        # Remove extra space from the text if any
        job_description_data = remove_extra_space(text=job_description_data)
        # Summarize the job description
        job_description_data = summarize_job_description(job_description=job_description_data)

        logger.info(f"[THREAD {thread_id}] JD parsed successfully (len={len(job_description_data)} chars)")

        # Cleanup
        if os.path.exists(temp_jd_path):
            os.unlink(temp_jd_path)
            logger.info(f"[THREAD {thread_id}] Deleted JD temp file")

        # Save to DB
        insert_job_description(thread_id=thread_id, job_description=job_description_data)

        return JSONResponse({"success": True, "message": "Job description stored"})

    except HTTPException:
        if temp_jd_path and os.path.exists(temp_jd_path):
            try:
                os.unlink(temp_jd_path)
            except:
                pass
        raise
    except Exception as e:
        logger.critical(f"[THREAD {thread_id}] Unexpected error in /upload_job_description: {str(e)}")
        if temp_jd_path and os.path.exists(temp_jd_path):
            try:
                os.unlink(temp_jd_path)
            except:
                pass
        raise HTTPException(status_code=500, detail="Failed to process job description")

# ====================== Global Task Tracker ======================
tasks = {}  # {task_id: dict}

# ====================== SSE Progress Endpoint ======================
@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    async def event_stream():
        while True:
            if task_id not in tasks:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                break

            task = tasks[task_id]
            yield f"data: {json.dumps(task)}\n\n"

            if task["status"] in ["completed", "failed"]:
                del tasks[task_id]
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

# ====================== Start Resume Upload Task ======================
@app.post("/upload_resume")
async def upload_files(
    background_tasks: BackgroundTasks,
    resume_files: List[UploadFile] = File(...),
    thread_id: str = Depends(check_thread_id)
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "total": 0,
        "current": 0,
        "message": "Starting upload...",
        "resume_count": len(resume_files)
    }

    background_tasks.add_task(process_resumes, task_id, resume_files, thread_id)
    return JSONResponse({"task_id": task_id, "status": "started"})

# ====================== Background Resume Processing ======================
async def process_resumes(task_id: str, resume_files: List[UploadFile], thread_id: str):
    task = tasks[task_id]  # Get reference once
    task["message"] = "Initializing AI model..."
    task["progress"] = 5
    await asyncio.sleep(0.1)

    try:
        # === Initialize variables ===
        success = {"resumes": [], "job_description": False}
        processed_count = 0
        failed_analyses = []

        # === Fetch JD ===
        job_description_data = get_job_description(thread_id=thread_id)
        if not job_description_data:
            raise HTTPException(status_code=400, detail="Job description not found...")
        
        success["job_description"] = True  # Mark JD as present

        # === Validate inputs ===
        if not resume_files or not any(f.filename for f in resume_files):
            raise HTTPException(status_code=400, detail="At least one resume file is required.")

        total_files = len([f for f in resume_files if f.filename])
        task["total"] = total_files
        logger.info(f"[THREAD {thread_id}] Processing {total_files} resumes (task: {task_id})")

        

        # === Process each resume ===
        for idx, resume_file in enumerate(resume_files):
            if not resume_file.filename:
                continue

            current = sum(1 for f in resume_files[:idx+1] if f.filename)
            task["current"] = current
            task["progress"] = 5 + int((current / total_files) * 90)
            task["message"] = f"Processing {resume_file.filename} ({current}/{total_files})"
            await asyncio.sleep(0.1)

            temp_resume_path: str | None = None
            try:
                ext = resume_file.filename.rsplit(".", 1)[-1].lower()
                if ext not in {"pdf", "txt", "docx"}:
                    raise ValueError(f"Unsupported file type: .{ext}")

                raw_bytes = await resume_file.read()
                if not raw_bytes:
                    raise ValueError("Empty file")

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
                temp_file.write(raw_bytes)
                temp_file.close()
                temp_resume_path = temp_file.name

                # Parse
                logger.info(f"Parsing resume {idx} ...")
                parsed_resume_text = parse_file(temp_resume_path)
                logger.info(f"Parsing done for resume {idx} ...")


                # Remove extra spaces from resume 
                parsed_resume_text = remove_extra_space(text=parsed_resume_text)

                logger.info(f"Extracting data from  resume {idx} ...")
                # Extract the data from resume
                extracted_resume_data = extract_data_from_resume(resume_data=parsed_resume_text)

                # Analyze
                analysis_result = analyze_resume(
                    extracted_resume_data=extracted_resume_data,
                    job_description=job_description_data
                )
                fit_score = int(analysis_result['fit_score'])
                resume_analysis_summary = analysis_result['resume_analysis_summary']


                if fit_score > 7:
                    hire = 1
                    extracted_resume_data.update({
                        "shortlisted" : 1
                    })
                else:
                    hire = 0
                    extracted_resume_data.update({
                        "shortlisted" : 0
                    })
                    
                hire_probability = AI.predict_hiring_decision(
                    resume_text=parsed_resume_text,
                    job_description=job_description_data
                )['Probability']

                AI.improve_model(
                    resume_text=parsed_resume_text,
                    job_description=job_description_data,
                    true_label=hire
                )

                extracted_resume_data.update({
                    "fit_score": fit_score,
                    "resume_analysis_summary": resume_analysis_summary,
                    "thread_id": thread_id,
                    "ai_hire_probability": hire_probability,
                    "evaluated": 0
                })

                # # check if the extracted email , linked in url and phone number are valid or not

                # logger.info(f"Starting check for info")
                # email_address , linkedin_url , phone_number = validate_contact_info(
                #     email=extracted_resume_data['email_address'],
                #     linkedin_url=extracted_resume_data['linkedin_url'],
                #     phone_number=extracted_resume_data['contact_number']
                # )


                # logger.info("Check done..")

                # extracted_resume_data.update({
                #     "email_address" : email_address,
                #     "linkedin_url" : linkedin_url,
                #     "phone_number" : phone_number
                # })


                


                insert_extracted_data(extracted_resume_data=extracted_resume_data)
                processed_count += 1

                success["resumes"].append({
                    "filename": resume_file.filename,
                    "parsed": True,
                    "length": len(parsed_resume_text)
                })

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

        if processed_count == 0:
            raise ValueError("No resumes were successfully processed.")

        # === Final Steps ===
        task["progress"] = 95
        task["message"] = "Finding fittest candidates..."
        await asyncio.sleep(0.1)

        fittest_candidates = get_fittest_candidates(thread_id=thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [],
            "thread_id": thread_id,
            "sql_retrieval": ""
        }
        workflow.update_state(config=config, values=initial_state)

        # === Mark Complete ===
        task["status"] = "completed"
        task["progress"] = 100
        task["message"] = "Complete!"
        task["data"] = {
            "success": True,
            "uploaded": success,
            "both_uploaded": processed_count > 0 and success["job_description"],
            "resume_count": total_files,
            "processed_count": processed_count,
            "failed_count": len(failed_analyses),
            "failed_analyses": failed_analyses or None,
            "fittest_candidates": fittest_candidates
        }

        logger.info(f"[THREAD {thread_id}] Upload complete: {processed_count} processed")

    except Exception as e:
        # Ensure task is marked failed and cleaned up
        if task_id in tasks:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = str(e)
            tasks[task_id]["progress"] = 0
        logger.error(f"[THREAD {thread_id}] Task {task_id} failed: {str(e)}")


@app.post("/upload_files")
async def upload_files(
    thread_id: str = Depends(check_thread_id),
    files: List[UploadFile] = File(...)  # ← plural + List
):
    temp_paths = []
    try:
        if not files or not any(f.filename for f in files):
            raise HTTPException(status_code=400, detail="At least one file is required.")

        processed_count = 0
        for file in files:
            if not file.filename:
                continue

            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in {"pdf", "txt", "docx", "csv", "json", "xlsx", "pptx"}:
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue

            raw_bytes = await file.read()
            if not raw_bytes:
                continue

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            temp_file.write(raw_bytes)
            temp_file.close()
            temp_path = temp_file.name
            temp_paths.append(temp_path)

            file_document = parse_file(temp_path, parsing_for_vector=True)
            vectorStore.store_embeddings(
                thread_id=thread_id,
                documents=file_document
            )
            processed_count += 1

            logger.info(f"[THREAD {thread_id}] {file.filename} File added to knowledge base ")
        return JSONResponse({
            "success": True,
            "message": f"Successfully uploaded {processed_count} file(s)",
            "processed_count": processed_count
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[THREAD {thread_id}] File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="File processing failed")
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

# ====================== Query Endpoint ======================
@app.post('/query')
def answer_query(
    thread_id: str = Depends(check_thread_id),
    query: str = Form(...)
):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        saved_state = workflow.checkpointer.get(config)
        if not saved_state:
            raise HTTPException(status_code=404, detail="Thread not found. Upload JD & resumes first.")

        input_state = {
            "thread_id": thread_id,
            "messages": [HumanMessage(content=query)]
        }

        result = workflow.invoke(input_state, config=config)
        ai_message = result["messages"][-1]

        response_content = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)
        return JSONResponse({"response": response_content or "No response generated."})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed for thread {thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query.")

# ====================== Health Check ======================
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ====================== Run Server ======================
if __name__ == "__main__":
    
    logger.info(f"GROQ API key :- {os.environ['GROQ_API_KEY']}")
    
    os.makedirs(os.environ.get('DATABASE_DIR', ''), exist_ok=True)
    os.makedirs(os.environ.get('CHAT_HISTORY_DIR', ''), exist_ok=True)
    os.makedirs(os.environ.get("EMBEDDING_DIR",'') , exist_ok=True)

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