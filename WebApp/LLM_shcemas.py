# schemas.py
from pydantic import BaseModel, Field 
from typing import Optional , Literal , TypedDict , Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id : str
    analyzed_resume_data : str
    job_description : str
    sql_retrieval : str

class ExtractResumeDataSchema(BaseModel):
    candidate_name : Optional[str] = Field(None , description="Name of the candidate.")
    email_address: Optional[str] = Field(None, description="Email address")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn URL")
    total_experience: int = Field(..., description="Total years of experience", ge=0)
    skills: str = Field(..., description="List of all skills")
    education: str = Field(..., description="Education summaries")
    work_experience: str = Field(..., description="Work experience summaries")
    projects: str = Field(..., description="Project summaries")


class ClassifyQuery(BaseModel):
    sentiment : Literal['hr','general'] = Field(... , description="classify the query of user if it is related to human resource related or not.")

class ResumeAnalysis(BaseModel):
    fit_score : int = Field(... , description="A fit score of the candidate.")
    analysis_summary : str = Field(... , description="A short and concise summary of analysis")
