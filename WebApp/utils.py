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
from LLM_models import llm_resume_data_extractor , llm_resume_analysis , chat_llm
from tenacity import retry , stop_after_attempt , wait_fixed
from database_sqlite import get_non_evluated_candidates , update_evaluated_candidates


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
    
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
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
    
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def analyze_resume(extracted_resume_data: dict, job_description: str):
    # Extract data from the resume dictionary
    
    total_experience = int(extracted_resume_data.get("total_experience", 0))
    skills = extracted_resume_data.get("skills", [])
    work_experience = extracted_resume_data.get("work_experience", "")
    projects = extracted_resume_data.get("projects", "")

    # Format the extracted resume data into a structured string for the model
    formatted_extracted_data = f"""
    # Total Experience:
    - {total_experience} years

    # Skills:
    - {', '.join(skills) if skills else "None provided"}

    # Work Experience:
    - {work_experience if work_experience else "No work experience provided"}

    # Projects:
    - {projects if projects else "No project details provided"}
    """

    # Create the prompt for the language model
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a senior technical recruiter and resume screening expert, with extensive experience in high-volume hiring for Fortune 500 companies. 
        Only evaluate based on explicit evidence in the resume and job description.
        """),

        ("human", f"""
        # JOB DESCRIPTION
        {job_description}

        # EXTRACTED RESUME DATA
        {formatted_extracted_data}

        ## YOUR TASK:
        Perform a deep analysis of the extracted resume data against the job description:- 
        - Calculate a fit score from 0 to 10 based on how well the candidate aligns with the job requirements 
        (like if skills mentioned , total years of experience is greater than or equal , previous work experience is good enough to match the requirement in job description ,
        It is fine if only few skills do not match , look for key skills required for the job role in job description and experience of candidate with the past work experience to give fit score.

        - Provide a short summary of your analysis, detailing the specific criteria and reasoning you used to evaluate the candidate.
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
        "resume_analysis_summary": analysis_result.resume_analysis_summary
    }

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def get_fittest_candidates(thread_id: str) -> str:
    """
    Return a formatted string of analyzed and shortlisted candidates by AI .
    """
    try:
        non_evaluated_candidates = get_non_evluated_candidates(thread_id=thread_id)

        if not non_evaluated_candidates:
            
            # Not canidates found fit score greater than 7 
            prompt = ChatPromptTemplate.from_template(
                '''
                You are a shortlisting HR AI assisstant but you didn't found any candidate fit for given 
                job description , generate a very short message saying that none of the candidates are fit 
                for the given job description . 
                
                Follow-Up :- Provide one liner follow up question which user can ask you in return.
                '''
            )

            chain = prompt | chat_llm | StrOutputParser()

            response = chain.invoke({})

            return response

        # Start building formatted string
        result = ["FITTEST CANDIDATES"]
        result.append("═" * 60)


        all_evaluated_tid = []
        for idx, row in enumerate(non_evaluated_candidates, start=1):
            tid,name, email, exp, score, analysis , shorlisted = row

            shorlisted = int(shorlisted)

            # Clean and format
            exp_str = f"{exp} year{'s' if exp != 1 else ''}" if exp else "N/A"

            candidate_block = [
                f"{idx}. Name: {name}",
                f"   Email: {email}",
                f"   Experience: {exp_str}",
                f"   Fit Score: {score:.1f}",
                f"   Analysis: {analysis.strip() if analysis else 'No analysis'}"
                f"   Shorlisted: {"Candidate is selected." if shorlisted else 'Candidate is rejected'}"
            ]
            result.extend(candidate_block)
            result.append("")  # blank line between candidates
            all_evaluated_tid.append(tid)
        

        result.append("═" * 60)
        result.append(f"Total: {len(non_evaluated_candidates)} candidate{'s' if len(non_evaluated_candidates) != 1 else ''} above threshold.")
        
        formatted_candidate_data = "\n".join(result)

        ###########   Langchain workflow   ############


        prompt = ChatPromptTemplate.from_template(
            '''
            You are an expert HR analyst. Below is a list of detailed of all candidates details who applied for a job.
            The fit score is assigned to the candidate to denote how fit they are for the job role assigned.
            Ideally only those candidates are shortliested whose fit score is greater than 7 for analysis , but if there
            aren't any shortlisted candidates then respond with approriate details of candidate which has some potential based on the analysis details that
            you have , but if none of the candidate are good enough then simply reply with apporiate response that no candidates are fit for the given job desription.


            **Candidate Data:**
            {candidate_data}

            ---

            Write a **concise 3–4 sentence summary** of only candidates that were selected by you highlighting:Name of candidates , email address ,fit score , Key strengths (e.g., experience, skills)

            Follow-Up :- Provide one liner follow up question which user can ask you in return , based on the data that you have 

            '''
        )

        chain = prompt | chat_llm | StrOutputParser()

        response = chain.invoke({
            "candidate_data": formatted_candidate_data
        }).strip()


        # After the response is generated update the users table to mark
        # candidates as evaluated.

        update_evaluated_candidates(
            thread_id=thread_id,
            tid_list=all_evaluated_tid
        )
        
        return response

    except Exception as e:
        error_msg = f"Error in get_fittest_candidates: {e}"
        print(error_msg)
        return error_msg
