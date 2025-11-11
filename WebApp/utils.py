# utils.py
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.prompts import ChatPromptTemplate
from LLM_models import llm_resume_data_extractor , llm_sentiment_analyzer , llm_resume_analysis
from typing import Literal
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
        - Name of the candidate if provided.
        - Total years of experience.
        - A list of all skills mentioned in the resume.
        - The candidate's email address (if mentioned).
        - The candidate's LinkedIn profile URL.
        - A summary of the candidate's education, including degrees, GPA (if available), and institution names.
        - A brief summary of the candidate's work experience.
        - A summary of notable projects, aside from work experience, that the candidate has worked on.
        """)
    ])

    # Execute the chain and invoke
    data_extraction_chain = analysis_prompt | llm_resume_data_extractor
    result = data_extraction_chain.invoke(input={"resume_data": resume_data})

    return  {
            "candidate_name" : result.candidate_name,
            "email_address": result.email_address,  
            "linkedin_url": result.linkedin_url,  
            "total_experience": result.total_experience, 
            "skills": result.skills, 
            "education": result.education,  
            "work_experience": result.work_experience,
            "projects": result.projects
        }


def classify_query(query: str) -> Literal["hr", "general"]:
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant specializing in recruitment and human resources.

        Your task is to classify the following user query into one of two categories:
        - hr :- The query is related to hiring, recruitment, jobs, human resources, onboarding, employee management, benefits, payroll, or related HR topics.
        - general :- The query is about anything else not related to HR.

        Return only one word as your answer: either "hr" or "general".
        - User might ask for analysis , providing the rat

        User Query:
        {user_query}
        """
    )
    classify_chain = prompt | llm_sentiment_analyzer

    result = classify_chain.invoke({"user_query": query})
    return result
    

def analyze_resume(extracted_resume_data: dict, job_description: str):
    # Extract data from the resume dictionary
    candidate_name = extracted_resume_data.get("candidate_name", "")
    email_address = extracted_resume_data.get("email_address", "")
    linkedin_url = extracted_resume_data.get("linkedin_url", "")
    total_experience = int(extracted_resume_data.get("total_experience", 0))
    skills = extracted_resume_data.get("skills", [])
    education = extracted_resume_data.get("education", "")
    work_experience = extracted_resume_data.get("work_experience", "")
    projects = extracted_resume_data.get("projects", "")

    # Format the extracted resume data into a structured string for the model
    formatted_extracted_data = f"""
    # Candidate Information:
    - Name: {candidate_name}
    - Email: {email_address}
    - LinkedIn: {linkedin_url}
    # Total Experience:
    - {total_experience} years

    # Skills:
    - {', '.join(skills) if skills else "None provided"}

    # Education:
    - {education if education else "No education details provided"}

    # Work Experience:
    - {work_experience if work_experience else "No work experience provided"}

    # Projects:
    - {projects if projects else "No project details provided"}
    """

    # Create the prompt for the language model
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a senior technical recruiter and resume screening expert, with extensive experience in high-volume hiring for Fortune 500 companies. 
        You are extremely rigorous, data-driven, and unflinchingly honest in your evaluation. 
        Never sugarcoat weaknesses and never assume unstated skills. 
        Only evaluate based on explicit evidence in the resume and job description.
        """),

        ("human", f"""
        # JOB DESCRIPTION
        {job_description}

        # EXTRACTED RESUME DATA
        {formatted_extracted_data}

        ## YOUR TASK:
        1. Perform a forensic-level analysis of the extracted resume data against the job description.
        2. Calculate a fit score from 0 to 100 based on how well the candidate aligns with the job requirements.
        3. Provide a short summary of your analysis, detailing the specific criteria and reasoning you used to evaluate the candidate.
        4. Conclude with a brief opinion on how well the candidate fits the job.
        """)
    ])

    # Run the analysis
    analysis_chain = analysis_prompt | llm_resume_analysis

    # Invoke the analysis and extract the result
    analysis_result = analysis_chain.invoke(input={
        "job_description": job_description,
        "extracted_resume_data": formatted_extracted_data
    })

    # Return the analysis results
    return {
        "fit_score": analysis_result.fit_score,
        "analysis_summary": analysis_result.analysis_summary
    }

