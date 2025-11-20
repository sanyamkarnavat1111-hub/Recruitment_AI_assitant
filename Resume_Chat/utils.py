
import re
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)
from llm import llm_document_classifier , llm_resume_data_extractor
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, wait_fixed , stop_after_attempt 
import os

def remove_extra_space(text: str) -> str:
    # Remove extra blank lines (more than one consecutive newline)
    cleaned_text = re.sub(r'\n+', '\n', text.strip())
    # Remove extra spaces between words (replace multiple spaces with a single space)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Ensure there is a single space after periods and punctuation
    cleaned_text = re.sub(r'([.,;!?])\s*', r'\1 ', cleaned_text)
    return cleaned_text.strip()


def parse_file(file_path: str , parsing_for_vector=False) -> str:
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
        elif extension == ".csv":
            loader = CSVLoader(file_path=file_path)
        else:
            print(f"Unsupported extension: {extension}")
            return ""

        documents = loader.load()
        if not documents:
            print(f"No content extracted from {file_path}")
            return ""
        
        # If the files besides job description or resume is uploaded
        # via the upload button in chat input window then return 
        # the documents so that they can be converted to vectors
        if parsing_for_vector:
            return documents
        
        # Combine all pages into one clean string
        full_text = "\n\n".join(doc.page_content for doc in documents)
        return full_text.strip()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""
    
@retry(wait=wait_fixed(5) , stop=stop_after_attempt(5))
def classify_document(text : str):
    
    document_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert document classifier specialized in recruitment workflows.
    Your task: classify the uploaded document into exactly ONE of these three categories:

    1. RESUME → Candidate's CV/resume (contains name, experience, skills, education, projects, contact info, etc.)
    2. JOB_DESCRIPTION → Job posting / JD (contains role title, responsibilities, requirements, qualifications, company info, salary, location, etc.)
    3. GENERAL → Anything else (cover letter, contract, policy, random text, invoice, etc.)

    Rules:
    - Look for strong signals:
    • RESUME: words like "Experience", "Skills", "Education", "Projects", years (20XX–20XX), phone/email at top, "Summary", "Achievements"
    • JOB_DESCRIPTION: words like "We are hiring", "Responsibilities", "Requirements", "Qualifications", "What you'll do", "Benefits", "Apply now", salary range
    - If it has both (e.g., resume with JD pasted), choose the PRIMARY intent — most uploads are pure.
    - NEVER explain, never hedge, never say "could be".
    - Output ONLY the exact label: RESUME or JOB_DESCRIPTION or GENERAL

    Examples:
    Input → "John Doe ... Software Engineer ... 5 years in Python ..."
    Output → RESUME

    Input → "Senior Data Scientist ... Responsibilities: ... Requirements: Master's degree ..."
    Output → JOB_DESCRIPTION

    Input → "Dear hiring manager, I am writing to apply ..."
    Output → GENERAL
    """),

        ("human", "Classify this document:\n\n{text}\n\nLabel:" )
    ])
    
    chain = document_classifier_prompt | llm_document_classifier

    result = chain.invoke(input={
        "text" : text
    })

    return result.document_type
    

@retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
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
        - Name of the candidate if present.
        - Job role which the resume belongs to (Can be of any profession)
        - Contact or phone number present . 
        - Location if present.
        - Total years of experience(Calculate exact years of professional experience if multiple).
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
            "job_role" : result.job_role,
            "contact_number" : result.contact_number,
            "location" : result.location,
            "email_address": result.email_address,  
            "linkedin_url": result.linkedin_url,  
            "total_experience": result.total_experience, 
            "skills": result.skills, 
            "education": result.education,  
            "work_experience": result.work_experience,
            "projects": result.projects
        }
    

if __name__ == "__main__":

    resume_text_1 = parse_file(file_path="Uploads/app_developer_resume.docx")
    resume_text_2 = parse_file(file_path="Uploads/data_analyst_resume.pdf")
    job_description_text = parse_file(file_path="Uploads/job_description.docx")
    general_document_text = parse_file(file_path="Uploads/privacy-policy-template.docx")

    resume_1_classification = classify_document(text=resume_text_1)
    resume_2_classification = classify_document(text=resume_text_2)

    job_description_classification = classify_document(text=job_description_text)

    general_document_classification = classify_document(text=general_document_text)


    print(f"Resume text 1 :-" , resume_1_classification)
    print(f"Resume text 2 :-" , resume_2_classification)


    print(f"JD classification :-" , job_description_classification)

    print(f"General document classification :-" , general_document_classification)
