import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()


DB_PATH = f"{os.environ['DATABASE_DIR']}/Users.db"



def get_db_connection():
    """Create and return a new SQLite connection."""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print("Error returning database connection: ", str(e))
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
        print("Connection is working ...")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print("Failed to test the connection:", str(e))
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                thread_id TEXT NOT NULL,
                candidate_name TEXT,
                job_role TEXT,
                contact_number TEXT,
                location TEXT,
                email_address TEXT,
                linkedin_url TEXT,
                total_experience INTEGER NOT NULL,
                skills TEXT NOT NULL,
                education TEXT,
                work_experience TEXT,
                projects TEXT
            )
        ''')
        conn.commit()
        print("Users table created successfully.")

        cur.execute('''
        CREATE TABLE IF NOT EXISTS job_description (
            jid INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            job_desc TEXT NOT NULL        
        )
        ''')
        print("Job description table created successfully.")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Error creating table:", str(e))
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
        print("Dropped table successfully.")
    except Exception as e:
        conn.rollback()
        print("Error dropping table:", str(e))
    finally:
        conn.close()


def insert_extracted_data(
    extracted_resume_data : dict
):
    """Insert a new candidate record."""

    
    try:
        # Extract fields for DB
        thread_id = extracted_resume_data.get("thread_id", "")
        candidate_name = str(extracted_resume_data.get("candidate_name", "")).lower()
        job_role = str(extracted_resume_data.get("job_role", "")).lower()
        contact_number = extracted_resume_data.get("contact_number", "")
        location = str(extracted_resume_data.get("location", "")).lower()
        email_address = extracted_resume_data.get("email_address", "")
        linkedin_url = extracted_resume_data.get("linkedin_url", "")
        total_experience = int(extracted_resume_data.get("total_experience", 0))
        skills = extracted_resume_data.get("skills", [])
        education = extracted_resume_data.get("education", "")
        work_experience = extracted_resume_data.get("work_experience", "")
        projects = extracted_resume_data.get("projects", "")

        conn = get_db_connection()

        cur = conn.cursor()
        insert_query = '''
            INSERT INTO users (
                thread_id, candidate_name,job_role ,contact_number, location, email_address,
                linkedin_url, total_experience, skills, education, work_experience,
                projects
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,? )
        '''
        values = (
            thread_id, candidate_name, job_role , contact_number , location,
            email_address, linkedin_url,total_experience, skills,
            education, work_experience,projects
        )
        cur.execute(insert_query, values)
        conn.commit()
        print("Data inserted successfully!")
    except Exception as e:
        conn.rollback()
        print("Error inserting data:", str(e))
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
        print(f"Inserted job description for THREAD[{thread_id}]")
    except Exception as e :
        print("Error inserting job description ..." , e)
    finally : 
        cur.close()
        conn.close()