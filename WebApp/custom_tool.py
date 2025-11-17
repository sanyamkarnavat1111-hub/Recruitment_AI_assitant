from LLM_models import llm_sql_fixer , llm_sql_query_generator
from langchain_core.prompts import PromptTemplate
from database_sqlite import get_db_connection

class SQLAgent:
    def generate_sql_query(self, thread_id: str , chat_history : list[str] , user_query : str) -> str:
        try:
            prompt = PromptTemplate(template='''
            You will be given chat history, a user query, the names of tables in the database, and their columns. Your task is to analyze the conversation and understand what the user is asking. Based on this understanding, you need to generate an SQL query that retrieves the relevant data from the database.

            **Important:** You must behave like read-only database reader ONLY 'SELECT' is allowed .The SQL query **must** include the `thread_id` in the `WHERE` clause to ensure the query retrieves user-specific data  
                 
            ### Database Structure:
            Currently, there are two tables in the database:

            1. **`users` table** with the following columns:
                - `thread_id` (TEXT)
                - `candidate_name` (TEXT) (For querying this column use pattern matching with 'like' sql clause )
                - `contact_number` (TEXT) 
                - `location` (TEXT) (For querying this column use pattern matching with 'like' sql clause)
                - `email_address` (TEXT)
                - `linkedin_url` (TEXT)
                - `total_experience` (INTEGER)
                - `skills` (TEXT)
                - `education` (TEXT)
                - `work_experience` (TEXT)
                - `projects` (TEXT)
                - `fit_score` (INTEGER)
                - `resume_analysis_summary` (TEXT)
                - `ai_hire_probability` (REAL)
                - `evaluated` (INTEGER)
                - `shortlisted` (INTEGER)

            2. **`job_description` table** with the following columns:
                - `thread_id` (TEXT)
                - `job_desc` (TEXT)

            ### Instructions:
            1. Carefully review the **conversation history** to understand the context and what the user is asking for.
            2. Examine the **user query** to identify the specific data the user wants.
            3. Using the above context and the available tables and columns, generate a **valid SQL query**.
            4. The query must include the `thread_id` in the `WHERE` clause to filter data for the specific thread. For example, you can write the query as WHERE thread_id = {thread_id}.
            5. If possible always use pattern like matching for names and emails or other text related queries avoid exact matches this might not fit good.
            
            ### Current Conversation:
            {history}

            ### User Query:
            {user_query}

            ### Provided Thread ID:
            {thread_id}

            ---

            Please generate the SQL query now:
            ''',
            input_variables=["history" , "user_query" , "thread_id"]
            )

            sql_query_generator = prompt | llm_sql_query_generator
            output = sql_query_generator.invoke(input={
                "history" : chat_history,
                "user_query" : user_query,
                "thread_id" : thread_id
            })

            return output.sql_query
            
            
        except Exception as e:
            return f"Error generating SQL: {str(e)}"

    
    def sql_query_fixer(self,thread_id ,sql_query : str) -> str:
        
        prompt = PromptTemplate(template='''
            You are given a SQL query that was generated to retrieve data from a database. Your job is to verify 
            if read-ony query is generated which uses only 'SELECT' and whether the query includes a WHERE thread_id = {thread_id} clause. 
            If the query does not contain the 'WHERE thread_id' clause, you must add it, ensuring the query is still valid and retrieves data for the 
            given thread ID. Also check if the given query is syntactically correct or not and if not , correct it so that it
            becomes sqlite3 compatible sql query that can be run in python.

            If the SQL query is syntactically correct and includes the where clause for thread id then simply return the same query.

            ### SQL Query:
            {sql_query}

            ### Thread ID:
            {thread_id}

            Does the query include WHERE thread_id = {thread_id}? If not, please modify the query by adding the 
            correct WHERE thread_id = {thread_id} clause and return the fixed query.
            ''',
            input_variables=['sql_query' , 'thread_id'])
        
        chain = prompt | llm_sql_fixer

        output  = chain.invoke(
            input={
                "sql_query" : sql_query,
                "thread_id" : thread_id
            }
        )
        return output.sql_query_fixed

    
    def execute_sql_query(self, sql_query: str) -> str:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)

            output = cursor.fetchall()

            if not output:
                return f"No relevant data found in database"

            formatted_output = ""
            
            for row in output:
                formatted_output += "-" * 20 + "\n"
                # Convert each value in the row to string before joining
                formatted_output += " | ".join(str(value) for value in row) + "\n"
            
            return formatted_output

        except Exception as e:
            return f"Error executing SQL query: {str(e)}"


