from pydantic import BaseModel, Field 
from typing import Optional , TypedDict , Annotated , Literal


class ExtractResumeDataSchema(BaseModel):
    candidate_name : Optional[str] = Field(None , description="Name of the candidate.")
    job_role : str = Field(..., description="What is the ideal job role of the user based on data provided.")
    contact_number : Optional[str] = Field(None , description="Contact number or phone number")
    location : Optional[str] = Field(None , description="Current location")
    email_address: Optional[str] = Field(None, description="Email address")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn URL")
    total_experience: int = Field(..., description="Total years of experience", ge=0)
    skills: str = Field(..., description="List of all skills")
    education: str = Field(..., description="Education summaries")
    work_experience: str = Field(..., description="Work experience summaries")
    projects: str = Field(..., description="Project summaries")


class DocumentClassifier(BaseModel):
    document_type : Literal['resume','job_description','general'] = Field(... , description="Classification of text if it is general or job resume")

class ChatRequest(BaseModel):
    thread_id : str
    user_query : str

class CreateSqlQuery(BaseModel):
    sql_query : str = Field(... , description="A sqlite3 supported sql query with compulsory use of where clause with thread_id ")

class FixSqlQuery(BaseModel):
    sql_query_fixed : str = Field(... , description="Fixed sql query that includes where clause")

class RagQueryRewritter(BaseModel):
    rewritten_query : str = Field(... , description="Rewritten user query with meaningful information for accurate RAG retrieval.")
