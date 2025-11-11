from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from LLM_models import chat_llm



# Step 1: Create the database connection
db = SQLDatabase.from_uri('sqlite:///Database/Users.db')

# Step 2: Create the SQL agent
sql_agent_executor = create_sql_agent(
    llm=chat_llm,
    db=db,
    agent_type="tool-calling",  # Use "tool-calling" for modern LLMs with tool support
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True
)



if __name__ == "__main__":

    thread_id = "b6c827c1-c478-48cc-b7d9-3b030a3fae15"
    # Step 3: Test the agent with a query
    query = f'''Give me emails of candidates related to thread_id {thread_id} '''

    response = sql_agent_executor.invoke(query)

    # Step 4: Print the response
    print(response['output'])
