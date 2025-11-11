from database_sqlite import get_db_connection
from langchain.tools import tool
import sqlite3
from LLM_models import chat_llm as llm
from langchain_core.prompts import ChatPromptTemplate

# Function to get all data for a given thread_id
@tool
def get_all_data(thread_id: int):
    """
    Retrieves all data for a given `thread_id` from the database.

    Args:
        thread_id (int): The ID of the thread for which to retrieve all candidate data.

    Returns:
        list: A list of dictionaries containing all fields (candidate name, email, skills, etc.)
        for each candidate in the thread. Each dictionary represents one candidate.
        dict: If no data is found for the given `thread_id`, returns an error message.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT * FROM user_data WHERE thread_id = ?;
        """
        cursor.execute(query, (thread_id,))
        results = cursor.fetchall()

        if results:
            data = []
            for result in results:
                data.append({
                    'id': result[0],
                    'thread_id': result[1],
                    'candidate_name': result[2],
                    'email_address': result[3],
                    'linkedin_url': result[4],
                    'total_experience': result[5],
                    'skills': result[6],
                    'education': result[7],
                    'work_experience': result[8],
                    'projects': result[9],
                    'job_description': result[10],
                    'fit_score': result[11],
                    'analysis': result[12]
                })
            return data
        else:
            return {'error': 'No data found for the given thread ID'}

    except sqlite3.DatabaseError as e:
        print(f"Error while fetching data: {e}")
        return {'error': 'Database error'}
    finally:
        conn.close()


# Function to get all candidates with a fit_score above a minimum threshold
@tool
def get_high_fit_candidates(thread_id: int, min_fit_score: float):
    """
    Retrieves candidates from a given `thread_id` with a fit_score greater than or equal to `min_fit_score`.

    Args:
        thread_id (int): The ID of the thread for which to retrieve candidate data.
        min_fit_score (float): The minimum fit score for filtering candidates.

    Returns:
        list: A list of candidate data for candidates with a fit_score >= `min_fit_score`.
        dict: If no candidates match the criteria, an error message is returned.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT * FROM user_data WHERE thread_id = ? AND fit_score >= ?;
        """
        cursor.execute(query, (thread_id, min_fit_score))
        results = cursor.fetchall()

        if results:
            data = []
            for result in results:
                data.append({
                    'id': result[0],
                    'thread_id': result[1],
                    'candidate_name': result[2],
                    'email_address': result[3],
                    'linkedin_url': result[4],
                    'total_experience': result[5],
                    'skills': result[6],
                    'education': result[7],
                    'work_experience': result[8],
                    'projects': result[9],
                    'job_description': result[10],
                    'fit_score': result[11],
                    'analysis': result[12]
                })
            return data
        else:
            return {'error': 'No matching candidates found'}

    except sqlite3.DatabaseError as e:
        print(f"Error: {e}")
        return {'error': 'Database error'}
    finally:
        conn.close()


# Function to get the candidate's email address based on thread_id
@tool
def get_candidate_email(thread_id: int):
    """
    Retrieves the email addresses of all candidates for a specific `thread_id`.

    Args:
        thread_id (int): The ID of the thread for which to retrieve candidate email addresses.

    Returns:
        list: A list of email addresses for the given `thread_id`.
        str: An error message if no email addresses are found for the `thread_id`.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT email_address FROM user_data WHERE thread_id = ?;
        """
        cursor.execute(query, (thread_id,))
        results = cursor.fetchall()

        if results:
            return [result[0] for result in results]
        else:
            return 'No emails found for this thread ID'

    except sqlite3.DatabaseError as e:
        print(f"Error: {e}")
        return 'Database error'
    finally:
        conn.close()

tools = [get_all_data , get_high_fit_candidates , get_candidate_email ]

llm_with_tools = llm.bind_tools(tools=tools)


prompt = ChatPromptTemplate.from_messages([
    ('system' , "You are a very helpful sqlilte query retriever support agent."),
    ('human' ,"Can you give me all data related to the thread id {thread_id} ")
])


chain = prompt | llm_with_tools

output = chain.invoke(input={
    "thread_id" :'0894ad80-eb52-4edb-aa72-cf7a04e58b6a'
})


# print(output)


