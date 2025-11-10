import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
assert os.environ.get('EXTERNAL_DATABASE_URL'), "EXTERNAL_DATABASE_URL is missing"

def get_db_connection():
    """Create and return a new PostgreSQL connection."""
    try:
        return psycopg2.connect(os.environ['EXTERNAL_DATABASE_URL'])
    except Exception as e:
        print("Error connecting to database:", str(e))
        raise

def test_connection():
    """Test DB connectivity by creating & dropping a temp table."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS test (tempid INT)")
            cur.execute("DROP TABLE test")
        conn.commit()
        print("Connection is working ...")
        return True
    except Exception as e:
        conn.rollback()
        print("Failed to test the connection:", str(e))
        return False
    finally:
        conn.close()

def create_table():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users(
                    tid SERIAL PRIMARY KEY,
                    thread_id VARCHAR(10000) NOT NULL,
                    candidate_name VARCHAR(100),
                    email_address VARCHAR(10000),
                    linkedin_url VARCHAR(10000),
                    total_experience SMALLINT NOT NULL,
                    skills VARCHAR(10000) NOT NULL,
                    education VARCHAR(10000),
                    work_experience VARCHAR(10000),
                    projects VARCHAR(10000),
                    analysis VARCHAR(10000),
                    resume_data VARCHAR(20000),
                    job_description_data VARCHAR(20000)
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
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS users")
        conn.commit()
        print("Dropped table successfully.")
    except Exception as e:
        conn.rollback()
        print("Error dropping table:", str(e))
    finally:
        conn.close()

def insert_extracted_data(
    thread_id: str,
    candidate_name :str,
    email_address: str,
    linkedin_url: str,
    total_experience: int,
    skills: str,
    education: str,
    work_experience: str,
    projects: str,
    analysis: str,
    resume_data : str,
    job_description_data : str
):
    
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            insert_query = '''
                INSERT INTO users (
                    thread_id, candidate_name , email_address, 
                    linkedin_url,total_experience, skills,
                    education, work_experience, projects, analysis , resume_data , job_description_data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s , %s , %s , %s)
            '''
            values = (
                thread_id, candidate_name , email_address, linkedin_url,
                total_experience, skills,
                education, work_experience, projects, analysis,
                resume_data , job_description_data
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
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE thread_id = %s LIMIT 1", (thread_id,))
            row = cur.fetchone()  # fetch the first (and only) row
            return row if row else []  # will be None if no row exists
    except Exception as e:
        print("Error checking thread:", str(e))
        return None
    finally:
        conn.close()


def get_all_data():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users")
            data = cur.fetchall()
            return data
    except Exception as e:
        print("Error getting rows:", str(e))
    finally:
        conn.close()




def truncate():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE users")
        conn.commit()
        print("Table truncated.")
    except Exception as e:
        conn.rollback()
        print("Error truncating table:", str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    
    test_connection()
    # drop_table()
    # create_table()
    # truncate()
    print(get_all_data())
