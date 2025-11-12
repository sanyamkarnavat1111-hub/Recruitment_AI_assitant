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
        print("Error connecting to database:", str(e))
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
                thread_id TEXT NOT NULL,
                candidate_name TEXT,
                email_address TEXT,
                linkedin_url TEXT,
                total_experience INTEGER NOT NULL,
                skills TEXT NOT NULL,
                education TEXT,
                work_experience TEXT,
                projects TEXT,
                fit_score INTEGER NOT NULL,
                analysis TEXT,
                ai_hire_probability REAL NOT NULL
            )
        ''')
        conn.commit()
        print("Table created successfully.")
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

    # Extract fields for DB
    thread_id = extracted_resume_data.get("thread_id", "")
    candidate_name = extracted_resume_data.get("candidate_name", "")
    email_address = extracted_resume_data.get("email_address", "")
    linkedin_url = extracted_resume_data.get("linkedin_url", "")
    total_experience = int(extracted_resume_data.get("total_experience", 0))
    skills = extracted_resume_data.get("skills", [])
    education = extracted_resume_data.get("education", "")
    work_experience = extracted_resume_data.get("work_experience", "")
    projects = extracted_resume_data.get("projects", "")
    fit_score = extracted_resume_data.get("fit_score" , "")
    analysis = extracted_resume_data.get("analysis" , "")
    ai_hire_probability = extracted_resume_data.get("ai_hire_probability" , "")


    conn = get_db_connection()
    try:
        cur = conn.cursor()
        insert_query = '''
            INSERT INTO users (
                thread_id, candidate_name, email_address,
                linkedin_url, total_experience, skills, education, work_experience,
                projects,fit_score, analysis , ai_hire_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? , ?)
        '''
        values = (
            thread_id, candidate_name, email_address, linkedin_url,
            total_experience, skills, education, work_experience, 
            projects,fit_score, analysis , ai_hire_probability
        )
        cur.execute(insert_query, values)
        conn.commit()
        print("Data inserted successfully!")
    except Exception as e:
        conn.rollback()
        print("Error inserting data:", str(e))
    finally:
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
        print("Error checking thread:", str(e))
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
        print("Error getting rows:", str(e))
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
        print("Table truncated (all rows deleted).")
    except Exception as e:
        conn.rollback()
        print("Error truncating table:", str(e))
    finally:
        conn.close()


if __name__ == "__main__":
    # For testing purposes
    test_connection()
    create_table()

    # Uncomment to reset or test
    # drop_table()
    # truncate()
    all_rows = get_all_data()
    if all_rows:
        for row in all_rows:
            print("-" * 100)
            print("Fit score:", row[0])
            print("Analysis summary:", row[1])
            print("\n")
    else:
        print("No rows found !!!")
