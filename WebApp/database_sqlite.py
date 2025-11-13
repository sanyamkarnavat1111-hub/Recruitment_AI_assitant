import sqlite3
import os
from dotenv import load_dotenv
import logging


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




DB_PATH = f"{os.environ['DATABASE_DIR']}/Users.db"


def get_db_connection():
    """Create and return a new SQLite connection."""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        logger.error("Error returning database connection: ", str(e))
        raise



def test_connection():
    """Test DB connectivity by creating & dropping a temp table."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS test (tempid INTEGER)")
        cur.execute("DROP TABLE test")
        conn.commit()
        logger.info("Connection is working ...")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.info("Failed to test the connection:", str(e))
        return False
    finally:
        if conn:
            conn.close()


def create_table():
    """Create the users table if it doesn't exist."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                tid INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                candidate_name TEXT,
                contact_number TEXT,
                location TEXT,
                email_address TEXT,
                linkedin_url TEXT,
                total_experience INTEGER NOT NULL,
                skills TEXT NOT NULL,
                education TEXT,
                work_experience TEXT,
                projects TEXT,
                fit_score INTEGER NOT NULL,
                resume_analysis_summary TEXT,
                ai_hire_probability REAL NOT NULL,
                evaluated INTEGER NOT NULL,
                shortlisted INTEGER NOT NULL
            )
        ''')
        conn.commit()
        logger.info("Users table created successfully.")

        cur.execute('''
        CREATE TABLE IF NOT EXISTS job_description (
            jid INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            job_desc TEXT NOT NULL        
        )
        ''')
        logger.info("Job description table created successfully.")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error creating table:", str(e))
    finally:
        conn.close()


def drop_table():
    """Drop the users table."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS users")
        conn.commit()

        cur.execute("DROP TABLE IF EXISTS job_description")
        conn.commit()
        logger.info("Dropped table successfully.")
    except Exception as e:
        conn.rollback()
        logger.error("Error dropping table:", str(e))
    finally:
        conn.close()


def insert_extracted_data(
    extracted_resume_data : dict
):
    """Insert a new candidate record."""

    
    try:
        # Extract fields for DB
        thread_id = extracted_resume_data.get("thread_id", "")
        candidate_name = extracted_resume_data.get("candidate_name", "")
        contact_number = extracted_resume_data.get("contact_number", "")
        location = extracted_resume_data.get("location", "")
        email_address = extracted_resume_data.get("email_address", "")
        linkedin_url = extracted_resume_data.get("linkedin_url", "")
        total_experience = int(extracted_resume_data.get("total_experience", 0))
        skills = extracted_resume_data.get("skills", [])
        education = extracted_resume_data.get("education", "")
        work_experience = extracted_resume_data.get("work_experience", "")
        projects = extracted_resume_data.get("projects", "")
        fit_score = extracted_resume_data.get("fit_score" , "")
        resume_analysis_summary = extracted_resume_data.get("resume_analysis_summary" , "")
        ai_hire_probability = extracted_resume_data.get("ai_hire_probability" , "")
        evaluated = int(extracted_resume_data.get("evaluated" , 0))
        shorlisted = int(extracted_resume_data.get("shortlisted" , 0))

        conn = get_db_connection()

        cur = conn.cursor()
        insert_query = '''
            INSERT INTO users (
                thread_id, candidate_name, contact_number, location, email_address,
                linkedin_url, total_experience, skills, education, work_experience,
                projects,fit_score, resume_analysis_summary , ai_hire_probability , evaluated , shortlisted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,? , ? , ? , ? , ? )
        '''
        values = (
            thread_id, candidate_name, contact_number , location,
            email_address, linkedin_url,total_experience, skills,
            education, work_experience,projects,fit_score, resume_analysis_summary ,
            ai_hire_probability , evaluated , shorlisted
        )
        cur.execute(insert_query, values)
        conn.commit()
        logger.info("Data inserted successfully!")
    except Exception as e:
        conn.rollback()
        logger.error("Error inserting data:", str(e))
    finally:
        conn.close()


def insert_job_description(
        thread_id : str,
        job_description : str,
):
    
    try :

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
        INSERT INTO job_description(
            thread_id , job_desc
        ) VALUES (? , ?) 
        ''',(thread_id , job_description))
        
        conn.commit()
        logger.info(f"Inserted job description for THREAD[{thread_id}]")
    except Exception as e :
        logger.error("Error inserting job description ..." , e)

    finally : 
        cur.close()
        conn.close()


def get_non_evluated_candidates(thread_id : str ):
    try :
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT tid , candidate_name, email_address, total_experience, fit_score, resume_analysis_summary , shortlisted 
            FROM users 
            WHERE thread_id = ? AND evaluated = 0
            ORDER BY fit_score DESC
            """
        cursor.execute(query, (thread_id,))
        rows = cursor.fetchall()
        return rows
    except Exception as e :
        logger.error(f"[THREAD {thread_id}] Error getting job description ..." , e)
    finally : 
        cursor.close()
        conn.close()

def update_evaluated_candidates(thread_id: str, tid_list: list[int]):
    if not tid_list:
        logger.warning(f"[THREAD {thread_id}] Empty tid_list passed to update_evaluated_candidates.")
        return 0

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Create the correct number of placeholders for the tid_list
        placeholders = ', '.join(['?'] * len(tid_list))
        
        query = f"""
            UPDATE users
            SET evaluated = 1
            WHERE thread_id = ? AND evaluated = 0 AND tid IN ({placeholders})
        """

        # Bind parameters (thread_id first, then all tids)
        params = [thread_id] + tid_list

        cursor.execute(query, params)
        conn.commit()

        return 1

    except Exception as e:
        logger.error(f"[THREAD {thread_id}] Error updating evaluated candidates: {e}")
        return 0

    finally:
        cursor.close()
        conn.close()





def get_job_description(thread_id :str) -> str:
    try :
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT job_desc FROM job_description where thread_id = ? ",(thread_id ,))
        job_desc = cur.fetchone()[0]

        return job_desc if job_desc else ""
    except Exception as e :
        logger.error(f"[THREAD {thread_id}] Error getting job description ..." , e)
    finally : 
        cur.close()
        conn.close()

def get_thread_data(thread_id: str):
    """Retrieve a single user record by thread_id."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE thread_id = ?", (thread_id,))
        row = cur.fetchall()
        return row if row else []
    
    except Exception as e:
        logger.error("Error checking thread:", str(e))
        return None
    finally:
        conn.close()


def get_all_data():
    """Retrieve all users' fit_score and analysis."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT fit_score, analysis FROM users")
        data = cur.fetchall()
        return data
    except Exception as e:
        logger.error("Error getting rows:", str(e))
        return []
    finally:
        conn.close()


def truncate():
    """Delete all rows from the users table (SQLite alternative to TRUNCATE)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users")
        conn.commit()
        logger.info("Table truncated (all rows deleted).")
    except Exception as e:
        conn.rollback()
        logger.error("Error truncating table:", str(e))
    finally:
        conn.close()


if __name__ == "__main__":
    # For testing purposes
    test_connection()
    create_table()

    # Uncomment to reset or test
    # drop_table()
    # truncate()
    
    # insert_job_description(
    #     thread_id="sanyam",
    #     job_description="Sample job description"
    # )

    # print(get_job_description(thread_id="sanyam"))

    
    