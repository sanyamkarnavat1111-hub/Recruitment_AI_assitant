from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage , HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from LLM_models import chat_llm
from LLM_shcemas import ChatState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from sql_agent import sql_agent_executor
import os

from dotenv import load_dotenv
load_dotenv()



def database_retriever(state: ChatState) -> ChatState:
    
    thread_id = state.get("thread_id")

    short_conversation_history = state['messages'][-10:]
    conversation_history = ""
    
    for msg in short_conversation_history :

        if isinstance(msg , HumanMessage):
            conversation_history += f"Human -- {msg.content}"
        else:
            conversation_history += f"AI -- {msg.content}"
    
    prompt = f'''
    You are a secure HR database assistant. 
    Use the thread_id in the WHERE clause to avoid data leakage for filtering data and writing queries.
    
    # Thread ID (must use in SQL): {thread_id}

    # This is the job description for your reference

    --- Conversation history so far ---
    {conversation_history}
    --- End ---

    Return only factual data from the DB. 
    If nothing found, say: "No data found."
    '''

    result = sql_agent_executor.invoke({"input": prompt})
    output = result.get("output", result)

    # CRITICAL: Return original messages + new state
    return {
        "sql_retrieval": output,
        "messages": AIMessage(content=output)
    }

    

def query(state: ChatState) -> ChatState:
    job_description = state.get("job_description", "")
    sql_retrieval = state.get("sql_retrieval", "No additional database info.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful HR assistant.
        """),
        ("human", """
        Use this information to answer note that some or all information may not be relevant so answer honestly
        based on whatever information you have amd keep the answer short and concise:

         
        # Job Description:
        {job_description}

        
        # Information retreived from Database agent (Can be empty or non relevant so analyze carefully before referring and answering based on it.)
        {sql_retrieval}
         
         # These is the conversation history so far 

        {conversation_history}
        """)
    ])

    chain = prompt | chat_llm | StrOutputParser()
    response = chain.invoke({
        "job_description": job_description,
        "sql_retrieval": sql_retrieval,
        "conversation_history": state['messages']
    })

    return {"messages": [AIMessage(content=response)]}


# Create the graph
graph = StateGraph(ChatState)
graph.add_node("query", query)
graph.add_node("database_retriever" , database_retriever)



graph.add_edge(START, "database_retriever")
graph.add_edge("database_retriever" , "query")
graph.add_edge("query", END)

# Sqlite 3 database checkpointer



os.makedirs(os.environ['CHAT_HISTORY_DIR'],exist_ok=True)

db_conn = sqlite3.connect(f"{os.environ['CHAT_HISTORY_DIR']}/chat_history.db" , check_same_thread=False)
sqlite_memory = SqliteSaver(conn=db_conn)


workflow = graph.compile(checkpointer=sqlite_memory)