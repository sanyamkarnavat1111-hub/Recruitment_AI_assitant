from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import uvicorn
from bot_graph import workflow, llm_analyzer
from utils import parse_file, analyze_resume
from langchain_core.messages import HumanMessage
import logging
import asyncio
import tempfile
from typing import Dict, AsyncGenerator

# ====================== Logging Setup ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory thread store
thread_store: Dict[str, Dict] = {}
temp_files_tracker: Dict[str, str] = {}  # file_key -> temp_path

# ====================== Lifespan (Startup + Shutdown) ======================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # --- Startup ---
    logger.info("Application starting up...")
    asyncio.create_task(periodic_health_check())
    logger.info("Periodic internal health check started (every 30s).")

    yield  # App runs here

    # --- Shutdown ---
    logger.info("Application shutting down... Cleaning up temporary files.")
    for temp_path in temp_files_tracker.values():
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    temp_files_tracker.clear()
    logger.info("Cleanup complete.")


# ====================== FastAPI App with Lifespan ======================
app = FastAPI(title="RecruitmentAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = "static"
UPLOADS = "Uploads"
os.makedirs(STATIC, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)


# ====================== Background Health Checker ======================
async def periodic_health_check():
    """Internal task that calls /health every 30 seconds."""
    while True:
        try:
            health_response = health_check()
            logger.info(f"[INTERNAL HEALTH CHECK] {health_response}")
        except Exception as e:
            logger.error(f"[INTERNAL HEALTH CHECK FAILED] {str(e)}")
        await asyncio.sleep(30)


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


# ====================== Helper: Validate File ======================
def validate_upload_file(file: UploadFile, field_name: str):
    if not file or not file.filename:
        return None
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for {field_name}. Allowed: {', '.join(allowed_extensions)}"
        )
    if file.size > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{field_name} exceeds 10MB limit."
        )
    return ext


# ====================== Helper: Save to Temp File ======================
def save_to_tempfile(contents: bytes, suffix: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(contents)
        temp_file.close()
        temp_path = temp_file.name
        return temp_path
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e


# ====================== Upload Endpoint ======================
@app.post("/upload")
async def upload_files(
    resume_file: UploadFile = File(None),
    job_description_file: UploadFile = File(None),
    thread_id: str = Form(...),
):
    if not thread_id or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id is required and cannot be empty."
        )

    if thread_id not in thread_store:
        thread_store[thread_id] = {
            "resume_path": None,
            "jd_path": None,
            "resume_data": None,
            "jd_data": None,
            "analysis": None,
            "files_uploaded": False
        }
    
    thread = thread_store[thread_id]
    success = {"resume": False, "job_description": False}

    try:
        # ---------- Resume ----------
        if resume_file and resume_file.filename:
            ext = validate_upload_file(resume_file, "resume")
            if not ext:
                return JSONResponse({
                    "success": False,
                    "uploaded": success,
                    "both_uploaded": False,
                    "analysis": None,
                    "error": "Resume file is invalid or missing."
                })

            contents = await resume_file.read()
            if not contents:
                raise ValueError("Resume file is empty.")

            temp_path = save_to_tempfile(contents, suffix=ext)
            temp_files_tracker[f"resume_{thread_id}"] = temp_path

            thread["resume_path"] = temp_path
            thread["resume_data"] = parse_file(temp_path)
            success["resume"] = True
            logger.info(f"Resume uploaded (temp) for thread {thread_id}: {resume_file.filename}")

        # ---------- Job Description ----------
        if job_description_file and job_description_file.filename:
            ext = validate_upload_file(job_description_file, "job description")
            if not ext:
                return JSONResponse({
                    "success": False,
                    "uploaded": success,
                    "both_uploaded": False,
                    "analysis": None,
                    "error": "Job description file is invalid or missing."
                })

            contents = await job_description_file.read()
            if not contents:
                raise ValueError("Job description file is empty.")

            temp_path = save_to_tempfile(contents, suffix=ext)
            temp_files_tracker[f"jd_{thread_id}"] = temp_path

            thread["jd_path"] = temp_path
            thread["jd_data"] = parse_file(temp_path)
            success["job_description"] = True
            logger.info(f"JD uploaded (temp) for thread {thread_id}: {job_description_file.filename}")

        # ---------- Run analysis ----------
        if thread["resume_data"] and thread["jd_data"] and thread["analysis"] is None:
            try:
                logger.info(f"Starting analysis for thread {thread_id}")
                thread["analysis"] = analyze_resume(
                    resume_data=thread["resume_data"],
                    job_description=thread["jd_data"],
                    llm_analyzer=llm_analyzer
                )
                thread["files_uploaded"] = True
                logger.info(f"Analysis completed for thread {thread_id}")
            except Exception as e:
                logger.error(f"Analysis failed for thread {thread_id}: {str(e)}")
                thread["analysis"] = None
                thread["files_uploaded"] = False
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Resume analysis failed: {str(e)}"
                )

        return JSONResponse({
            "success": True,
            "uploaded": success,
            "both_uploaded": thread["files_uploaded"],
            "analysis": thread["analysis"]
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in /upload for thread {thread_id}: {str(e)}")
        return JSONResponse({
            "success": False,
            "uploaded": success,
            "both_uploaded": False,
            "analysis": None,
            "error": "An unexpected error occurred during upload."
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ====================== Query Endpoint ======================
@app.post("/query")
async def answer_query(
    thread_id: str = Form(...),
    query: str = Form(...),
):
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required.")
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if thread_id not in thread_store:
        raise HTTPException(status_code=404, detail="Thread not found. Please upload files first.")
    if not thread_store[thread_id]["files_uploaded"]:
        raise HTTPException(status_code=400, detail="Both files must be uploaded and analyzed first.")

    thread = thread_store[thread_id]
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=query)],
        "conversation_thread": thread_id,
        "analyzed_resume_data": thread["analysis"],
        "job_description": thread["jd_data"],
        "resume_data": thread["resume_data"]
    }

    try:
        result = workflow.invoke(input_state, config=config)
        ai_message = result["messages"][-1]
        response_content = ai_message.content or "No response generated."
        return JSONResponse({"response": response_content})
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ====================== Health Check ======================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "threads_active": len(thread_store),
        "temp_files": len(temp_files_tracker)
    }


# ====================== Run Server ======================
if __name__ == "__main__":
    port = 3333
    logger.info(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)