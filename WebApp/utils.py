# utils.py
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.prompts import ChatPromptTemplate


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
    

def analyze_resume(resume_data : str , job_description : str , llm_analyzer):

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

    - Total Professional Experience (in years): Count only full-time, paid roles. Internships, projects, or education do not count unless explicitly full-time employment. Round down to nearest year.
    - Explicitly Mentioned Technical & Soft Skills: List only skills **directly stated** in the resume. Do not infer from project descriptions or roles.
    - AI opinion should to in a summarized version of what you think of the candidate profile.
     
     These are things based on provided a 50 to 80 word opinion paragraph
     
    From the Job Description**, extract and list:
    - Required minimum years of experience
    - Must-have skills (explicitly stated or strongly implied with words like "required", "must", "essential")
    - Nice-to-have skills (mentioned but not mandatory)

    
    Score **only** on:
    - Experience match 
    - Must-have skill coverage 
    - Bonus: Nice-to-have skills 
    
    **Formula:**
     Score = (exp_match_pct × 0.4) + (must_have_match_pct × 0.5) + (nice_to_have_bonus × 0.1)
    
    You **must** choose one:
    - **QUALIFY** → Only if:
    1. Experience ≥ JD minimum
    2. ≥75% of **must-have** skills are explicitly present in resume
    3. Score ≥ 7.0
    - **REJECT** → If any of the above fail

     """)])

    analysis_result = llm_analyzer.invoke(analysis_prompt.format_messages(
        job_description=job_description,
        resume_data=resume_data
    ))

    # Create a detailed AI message with full analysis
    analysis_message = f"""
    ## Resume Analysis Complete

    Total Experience: {analysis_result.total_experience} years  \n
    Skills: {', '.join(analysis_result.skills)}  \n
    Fit Score: {analysis_result.score}/10  \n
    AI Opinion: {analysis_result.opinion} \n
    """.strip()

    return analysis_message