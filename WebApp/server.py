from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from bot_graph import workflow, llm_analyzer
from utils import parse_file, analyze_resume
from langchain_core.messages import HumanMessage
import logging

# ====================== Logging Setup ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== FastAPI App Setup ======================
app = FastAPI(title="RecruitmentAI")

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

# In-memory thread store
thread_store = {}  # thread_id -> dict

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

    # Initialise thread if missing
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
            try:
                ext = validate_upload_file(resume_file, "resume")
                if not ext:
                    return JSONResponse({
                        "success": False,
                        "uploaded": success,
                        "both_uploaded": False,
                        "analysis": None,
                        "error": "Resume file is invalid or missing."
                    })

                resume_path = os.path.join(UPLOADS, f"resume_{thread_id}{ext}")
                contents = await resume_file.read()
                if not contents:
                    raise ValueError("Resume file is empty.")

                with open(resume_path, "wb") as f:
                    f.write(contents)

                thread["resume_path"] = resume_path
                thread["resume_data"] = parse_file(resume_path)
                success["resume"] = True
                logger.info(f"Resume uploaded for thread {thread_id}: {resume_file.filename}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Resume upload failed for thread {thread_id}: {str(e)}")
                thread["resume_path"] = None
                thread["resume_data"] = None
                return JSONResponse({
                    "success": False,
                    "uploaded": success,
                    "both_uploaded": False,
                    "analysis": None,
                    "error": f"Resume upload failed: {str(e)}"
                }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # ---------- Job Description ----------
        if job_description_file and job_description_file.filename:
            try:
                ext = validate_upload_file(job_description_file, "job description")
                if not ext:
                    return JSONResponse({
                        "success": False,
                        "uploaded": success,
                        "both_uploaded": False,
                        "analysis": None,
                        "error": "Job description file is invalid or missing."
                    })

                jd_path = os.path.join(UPLOADS, f"jd_{thread_id}{ext}")
                contents = await job_description_file.read()
                if not contents:
                    raise ValueError("Job description file is empty.")

                with open(jd_path, "wb") as f:
                    f.write(contents)

                thread["jd_path"] = jd_path
                thread["jd_data"] = parse_file(jd_path)
                success["job_description"] = True
                logger.info(f"JD uploaded for thread {thread_id}: {job_description_file.filename}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"JD upload failed for thread {thread_id}: {str(e)}")
                thread["jd_path"] = None
                thread["jd_data"] = None
                return JSONResponse({
                    "success": False,
                    "uploaded": success,
                    "both_uploaded": False,
                    "analysis": None,
                    "error": f"Job description upload failed: {str(e)}"
                }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # ---------- Run analysis only when BOTH files are present ----------
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id is required."
        )

    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    if thread_id not in thread_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found. Please upload files first."
        )

    thread = thread_store[thread_id]
    if not thread["files_uploaded"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both resume and job description must be uploaded and analyzed before querying."
        )

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=query)],
        "conversation_thread": thread_id,
        "analyzed_resume_data": thread["analysis"],
        "job_description": thread["jd_data"],
        "resume_data": thread["resume_data"]
    }

    try:
        logger.info(f"Processing query for thread {thread_id}: {query[:50]}...")
        result = workflow.invoke(input_state, config=config)
        ai_message = result["messages"][-1]
        response_content = ai_message.content if ai_message.content else "No response generated."

        return JSONResponse({
            "response": response_content
        })

    except KeyError as e:
        logger.error(f"KeyError in workflow result for thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid response structure from AI workflow."
        )
    except Exception as e:
        logger.error(f"Query processing failed for thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

# ====================== Health Check ======================
@app.get("/health")
def health_check():
    return {"status": "healthy", "threads_active": len(thread_store)}

# ====================== Run Server ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)