from langchain.tools import tool
from LLM_models import chat_llm  # Assuming chat_llm is the LLM you are using
from database_sqlite import get_db_connection

# Tool for getting job description based on thread_id
@tool
def get_job_description(thread_id: str) -> str:
    """
    Fetch the job description for a given thread_id from the job_description table.
    
    Parameters:
    thread_id (str): The thread identifier to fetch the job description for.
    
    Returns:
    str: The job description for the thread.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT job_desc FROM job_description WHERE thread_id = ?", (thread_id,))
        job_desc = cursor.fetchone()
        if job_desc:
            return job_desc[0]  # Extract job description from the result
        else:
            return f"Job description not found for thread {thread_id}"
    except Exception as e:
        return f"Error retrieving job description: {str(e)}"

# Tool for getting all details of a candidate based on thread_id
@tool
def get_all_details(thread_id: str) -> str:
    """
    Fetch all candidate details for a given thread_id from the users table.
    
    Parameters:
    thread_id (str): The thread identifier to fetch candidate details for.
    
    Returns:
    str: A summary of the candidate details.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT candidate_name, contact_number, location, email_address, 
                   linkedin_url, resume_analysis_summary, shortlisted
            FROM users WHERE thread_id = ?""", (thread_id,)
        )
        candidate_details = cursor.fetchone()
        if candidate_details:
            return f"""
            Candidate: {candidate_details[0]}
            Contact: {candidate_details[1]}
            Location: {candidate_details[2]}
            Email: {candidate_details[3]}
            LinkedIn: {candidate_details[4]}
            Summary: {candidate_details[5]}
            Shortlisted: {candidate_details[6]}
            """
        else:
            return f"Candidate details not found for thread {thread_id}"
    except Exception as e:
        return f"Error retrieving candidate details: {str(e)}"

# Tool for retrieving specific candidate information (e.g., name, email, location, etc.)
@tool
def get_candidate_info(thread_id: str, info_type: str) -> str:
    """
    Fetch specific information for a given candidate based on thread_id and requested info type.
    
    Parameters:
    thread_id (str): The thread identifier to fetch candidate details for.
    info_type (str): The type of information to fetch (e.g., 'name', 'email', 'location').
    
    Returns:
    str: The requested information or an error message if not found.
    """
    info_map = {
        "name": "candidate_name",
        "email": "email_address",
        "location": "location",
        "contact": "contact_number",
        "linkedin": "linkedin_url",
        "summary": "resume_analysis_summary",
        "skills": "skills",
        "education": "education",
        "work_experience": "work_experience",
        "projects": "projects",
        "fit_score": "fit_score",
        "shortlisted": "shortlisted",
        "total_experience": "total_experience"
    }
    
    if info_type not in info_map:
        return f"Invalid information type: {info_type}. Please request one of the valid types."
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {info_map[info_type]} FROM users WHERE thread_id = ?""", (thread_id,)
        )
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return f"{info_type.capitalize()} not found for thread {thread_id}"
    except Exception as e:
        return f"Error retrieving {info_type}: {str(e)}"

# Tool for checking if a candidate is shortlisted based on thread_id
@tool
def is_shortlisted(thread_id: str) -> str:
    """
    Check whether a candidate is shortlisted based on the thread_id.
    
    Parameters:
    thread_id (str): The thread identifier to check if the candidate is shortlisted.
    
    Returns:
    str: Whether the candidate is shortlisted or not.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT shortlisted FROM users WHERE thread_id = ?", (thread_id,))
        shortlisted = cursor.fetchone()
        if shortlisted:
            return "Shortlisted" if shortlisted[0] == 1 else "Not shortlisted"
        else:
            return f"No candidate found for thread {thread_id}"
    except Exception as e:
        return f"Error checking shortlist status: {str(e)}"

# Tool for fetching candidate resume summary based on thread_id
@tool
def get_resume_summary(thread_id: str) -> str:
    """
    Fetch the resume summary for a given candidate based on thread_id.
    
    Parameters:
    thread_id (str): The thread identifier to fetch the resume summary for.
    
    Returns:
    str: The resume summary for the thread.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT resume_analysis_summary FROM users WHERE thread_id = ?", (thread_id,))
        resume_summary = cursor.fetchone()
        if resume_summary:
            return resume_summary[0]
        else:
            return f"Resume summary not found for thread {thread_id}"
    except Exception as e:
        return f"Error retrieving resume summary: {str(e)}"

# Tool for fetching job fit score for a candidate based on thread_id
@tool
def get_fit_score(thread_id: str) -> str:
    """
    Fetch the fit score for a candidate based on thread_id.
    
    Parameters:
    thread_id (str): The thread identifier to fetch the fit score for.
    
    Returns:
    str: The fit score for the thread.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT fit_score FROM users WHERE thread_id = ?", (thread_id,))
        fit_score = cursor.fetchone()
        if fit_score:
            return f"Fit score: {fit_score[0]}"
        else:
            return f"Fit score not found for thread {thread_id}"
    except Exception as e:
        return f"Error retrieving fit score: {str(e)}"

# Tool for fetching candidate's total experience
@tool
def get_total_experience(thread_id: str) -> str:
    """
    Fetch the total experience for a candidate based on thread_id.
    
    Parameters:
    thread_id (str): The thread identifier to fetch the total experience for.
    
    Returns:
    str: The total experience for the thread.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT total_experience FROM users WHERE thread_id = ?", (thread_id,))
        total_experience = cursor.fetchone()
        if total_experience:
            return f"Total Experience: {total_experience[0]} years"
        else:
            return f"Total experience not found for thread {thread_id}"
    except Exception as e:
        return f"Error retrieving total experience: {str(e)}"


# Function to handle tool calls from LLM response
def handle_tool_call(response):
    """
    Handles tool calls from the LLM response by invoking the relevant tools
    and gathering their output.

    Parameters:
    response (object): The response object from the LLM, containing tool call data.

    Returns:
    str: A formatted string with the outputs from the tool calls.
    """
    result = ""

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']

            try:
                # Invoking the appropriate tool based on the name
                if tool_name == "get_job_description":
                    output = get_job_description.invoke(tool_call).content
                elif tool_name == "get_all_details":
                    output = get_all_details.invoke(tool_call).content
                elif tool_name == "get_candidate_info":
                    output = get_candidate_info.invoke(tool_call).content
                elif tool_name == "is_shortlisted":
                    output = is_shortlisted.invoke(tool_call).content
                elif tool_name == "get_resume_summary":
                    output = get_resume_summary.invoke(tool_call).content
                elif tool_name == "get_fit_score":
                    output = get_fit_score.invoke(tool_call).content
                elif tool_name == "get_total_experience":
                    output = get_total_experience.invoke(tool_call).content
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

                # Formatting the result with separator lines
                result += "-" * 10 + "\n"
                result += f"Output from tool '{tool_name}':\n"
                result += output + "\n"
                result += "-" * 10 + "\n"
            
            except Exception as e:
                print(f"Error invoking tool '{tool_name}': {str(e)}")
                result += f"Error invoking tool '{tool_name}': {str(e)}\n"
    else:
        print("No tool calls found in response.")
        result = "No tool calls found in the LLM response.\n"

    return result


# Main function to integrate tools and LLM
if __name__ == "__main__":
    # Example thread ID for testing
    thread_id = '60306e06-822a-444a-a0c3-dc8bc488231f'

    # Binding tools to the LLM
    tools = [
        get_job_description,
        get_all_details,
        get_candidate_info,
        is_shortlisted,
        get_resume_summary,
        get_fit_score,
        get_total_experience
    ]
    llm_with_tools = chat_llm.bind_tools(tools=tools)

    # LLM invocation that suggests tool calls
    query = f" Give me names of the candidate and their contact info if they are shortlisted or not '{thread_id}'"
    response = llm_with_tools.invoke(query)

    # Handle and format the tool call outputs
    tool_call_output = handle_tool_call(response)

    # Print the final formatted output
    print(tool_call_output)
