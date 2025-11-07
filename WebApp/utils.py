# utils.py
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLM_models import chat_llm , llm_resume_data_extractor
from tenacity import retry , stop_after_attempt , wait_fixed

def parse_file(file_path: str) -> str:
    """
    Load PDF, DOCX, or TXT file and return plain text string.
    Uses robust LangChain loaders.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return ""

    extension = Path(file_path).suffix.lower()

    try:
        if extension == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"Unsupported extension: {extension}")
            return ""

        documents = loader.load()
        if not documents:
            print(f"No content extracted from {file_path}")
            return ""

        # Combine all pages into one clean string
        full_text = "\n\n".join(doc.page_content for doc in documents)
        return full_text.strip()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""
    
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def extract_data_from_resume(resume_data: str) -> dict:
    
    # Define the extraction prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a very skilled resume data extractor. Your job is to extract relevant details from resumes.
        """),

        ("human", """
        Please analyze the following resume and extract the information as per the provided schema.

        # CANDIDATE RESUME
        {resume_data}

        Ensure that you extract the following details:

        1. Total years of experience.
        2. A list of all skills mentioned in the resume.
        3. The candidate's email address (if mentioned).
        4. The candidate's LinkedIn profile URL.
        5. A summary of the candidate's education, including degrees, GPA (if available), and institution names.
        6. A brief summary of the candidate's work experience.
        7. A summary of notable projects, aside from work experience, that the candidate has worked on.
        """)
    ])

    # Execute the chain and invoke
    data_extraction_chain = analysis_prompt | llm_resume_data_extractor
    result = data_extraction_chain.invoke(input={"resume_data": resume_data})

    return  {
            "email_address": result.email_address,  
            "linkedin_url": result.linkedin_url,  
            "total_experience": result.total_experience, 
            "skills": result.skills, 
            "education": result.education,  
            "work_experience": result.work_experience,
            "projects": result.projects
        }
    

def analyze_resume(resume_data : str , job_description : str ):

    analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a senior technical recruiter and resume screening expert and veteran in high-volume hiring for Fortune 500 companies. 
    You are extremely rigorous, data-driven, and unflinchingly honest in your evaluation. 
    Never sugarcoat weaknesses. Never assume unstated skills. 
    Only evaluate based on explicit evidence in the resume and job description.
    """),

    ("human", """
    # JOB DESCRIPTION
    {job_description}

    # CANDIDATE RESUME
    {resume_data}
    
    ---

    ## YOUR TASK: Perform a forensic-level resume-to-JD match analysis
    
    Once you have completely analyzed the resume of the user provided a short summary of what
    data, insights you got from the data and an opinion on how well the candidate aligns with given job description. 
    """)])

    analysis_chain = analysis_prompt | chat_llm | StrOutputParser()


    analysis_result = analysis_chain.invoke(input={
        "job_description":job_description,
        "resume_data":resume_data
    })

    

    return analysis_result

