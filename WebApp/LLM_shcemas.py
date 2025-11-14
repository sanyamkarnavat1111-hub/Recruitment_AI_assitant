# schemas.py
from pydantic import BaseModel, Field 
from typing import Optional , TypedDict , Annotated , Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id : str
    job_description : str
    sql_retrieval : str
    rag_retrieval : str

class QueryRouter(BaseModel):
    decision : Literal['rag' , 'sql' , 'none'] = Field(... , description="Decide whether extra knowledge base retrieval is needed or not at all required.")

class ExtractResumeDataSchema(BaseModel):
    candidate_name : Optional[str] = Field(None , description="Name of the candidate.")
    contact_number : Optional[str] = Field(None , description="Contact number or phone number")
    location : Optional[str] = Field(None , description="Current location")
    email_address: Optional[str] = Field(None, description="Email address")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn URL")
    total_experience: int = Field(..., description="Total years of experience", ge=0)
    skills: str = Field(..., description="List of all skills")
    education: str = Field(..., description="Education summaries")
    work_experience: str = Field(..., description="Work experience summaries")
    projects: str = Field(..., description="Project summaries")

class ResumeAnalysis(BaseModel):
    fit_score : int = Field(... , description="A fit score of the candidate.")
    resume_analysis_summary : str = Field(... , description="A short and concise summary of analysis")

class CreateSqlQuery(BaseModel):
    sql_query : str = Field(... , description="A sqlite3 supported sql query with compulsory use of where clause with thread_id ")

class FixSqlQuery(BaseModel):
    sql_query_fixed : str = Field(... , description="Fixed sql query that includes where clause")


