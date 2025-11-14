from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from LLM_models import chat_llm , chat_llm_ollama
from dotenv import load_dotenv
import os
from tenacity import retry , wait_fixed , stop_after_attempt

load_dotenv()


# Step 1: Create the database connection
os.makedirs(os.environ['DATABASE_DIR'] , exist_ok=True)

db = SQLDatabase.from_uri(f'sqlite:///{os.environ['DATABASE_DIR']}/Users.db')




@retry(stop=stop_after_attempt(3) , wait=wait_fixed(1))
def get_info_from_database(query :str):
    try:
        # Create the SQL agent
        sql_agent_executor = create_sql_agent(
            llm=chat_llm,
            db=db,
            agent_type="tool-calling",  # Use "tool-calling" for modern LLMs with tool support
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
            early_stopping_method="force",
        )

        result = sql_agent_executor.invoke({"input": query})
        output = result['output']
        return output
    except Exception as e :
        raise



if __name__ == "__main__":

    thread_id = "60306e06-822a-444a-a0c3-dc8bc488231f"
    # Step 3: Test the agent with a query
    query = f"give me entire job description for this thread_id {thread_id} from job description table ?"
    
    try:
        response = get_info_from_database(query)
        # Step 4: Print the response
        print(response)
    except:
        print("Database retriever failed.")

    
