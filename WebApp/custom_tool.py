from LLM_models import llm_sql_fixer , llm_sql_query_generator
from langchain_core.prompts import PromptTemplate
from database_sqlite import get_db_connection
from tenacity import retry , wait_exponential , stop_after_attempt

class SQLAgent:

    @retry(stop=stop_after_attempt(3) , wait=wait_exponential(multiplier=2 , max=20))
    def generate_sql_query(self, thread_id: str , chat_history : list[str] , user_query : str) -> str:
        try:
            prompt = PromptTemplate(template='''
            You are an assistant that generates SQLite3-compatible SELECT queries based on a conversation between a human and an AI.
            
            ### Your tasks:
            - Analyze the conversation history and the latest user query.
                - Identify meaningful details such as names, email addresses, or pronouns.

            - Use these extracted details to generate the most accurate and relevant SQL query possible.

            ### You are given:
                - The full chat history
                - The latest user message
                - A list of database tables
                - The columns inside those tables

            ### Important rules:
            - You are a read-only database agent. Only SELECT queries are allowed. No INSERT, UPDATE, DELETE, DROP, or modifications.
            - Every SQL query must include thread_id in the WHERE clause to ensure the result is specific to the correct user or conversation.
            
            ### If your query has to include columns candidate_name or email_address then :
                - Extract the best matching name or email from the conversation.
                - Convert the extracted name to lowercase using LOWER() in SQL.
                - Use pattern matching with LIKE to search for similar names or emails.
            
            Your final output should be only the SQL query, with no explanation.

            ### Database Structure:
            Currently, there are two tables in the database 'users' and 'job_description':

            1. 'users' table with the following columns:
                - thread_id (TEXT)
                - candidate_name (TEXT)
                - contact_number (TEXT) 
                - location (TEXT)
                - email_address (TEXT) 
                - linkedin_url (TEXT)
                - total_experience (INTEGER)
                - skills (TEXT)
                - education (TEXT)
                - work_experience (TEXT)
                - projects (TEXT)
                - fit_score (INTEGER)
                - resume_analysis_summary (TEXT)
                - ai_hire_probability (REAL)
                - evaluated (INTEGER)
                - shortlisted (INTEGER)

            2. job_description has with the following columns:
                - thread_id (TEXT)
                - job_desc (TEXT)

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

    @retry(stop=stop_after_attempt(3) , wait=wait_exponential(multiplier=2 , max=20))
    def sql_query_fixer(self,thread_id ,sql_query : str) -> str:
        
        prompt = PromptTemplate(template='''
            You are given a SQL query that was generated to retrieve data from a SQLite3 database.
                                
            Your job is to validate and correct this SQL query according to the following rules:

            ### 1. Read-Only Enforcement
            - The query must be strictly read-only and use ONLY the 'SELECT' statement.
            - If the query contains INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, REPLACE, or any non-SELECT operation, convert it into a valid SELECT query consistent with the intent of the original query.

            ### 2. Thread ID Requirement
            - The final SQL query MUST include a 'WHERE thread_id = {thread_id}' clause.  
            - If a WHERE clause already exists but does NOT include 'thread_id', append 'AND thread_id = {thread_id}'.
            - If there is no WHERE clause at all, add 'WHERE thread_id = {thread_id}'.
            - Never hallucinate a thread_id. Use the '{thread_id}' template variable exactly as provided.

            ### 3. Name / Email Pattern Matching
            - If the query includes filters on names or email addresses:
                - Convert any name or email to lowercase via SQL using LOWER(column_name).
                - Use pattern matching with LIKE to search for similar names or emails.

            ### 4. Syntax Validation (SQLite3 Compatible)
            - Validate whether the SQL syntax is correct for SQLite3.
            - If not correct, fix the syntax so the final SQL query runs in Python using sqlite3.
            - Ensure correct table/column reference formatting.
            - Remove trailing commas, mismatched parentheses, or repeated/duplicate columns.

            ### 5. Avoid Redundancy
            - Remove repeated columns, repeated WHERE conditions, or duplicated JOINs.
            - Keep the query minimal, clean, and logically consistent.

            ### 6. Output Requirements
            - If the SQL query is already syntactically valid, read-only, and includes the correct thread_id clause, return it unchanged.
            - Otherwise, return the corrected SQL query.
            - Return only the SQL query with no explanation.
            
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




if __name__ == "__main__":

    thread_id = "f3229e28-69c2-4e6e-a289-7498e115774c"
    chat_history = [
    "AI :- Following are the shortlisted candidates Kandace and Tomislav",
    "Human :- Ok great can you give their contact info ?",
    "AI : Here are their phone numbers 123124523 and 98578032412"
    ]

    user_query = f"Can you give me summarized info of these candidates and also how to contact them ? \n Thread ID :- {thread_id}"

    obj  = SQLAgent()


    generated_sql = obj.generate_sql_query(
        thread_id=thread_id,
        chat_history=chat_history,
        user_query=user_query
    )

    # fixed_sql_query = obj.sql_query_fixer(
    #     thread_id=thread_id,
    #     sql_query=generated_sql
    # )
    # print("Fixed SQL :-" , fixed_sql_query)


    print("Generated SQL :-" , generated_sql)
    