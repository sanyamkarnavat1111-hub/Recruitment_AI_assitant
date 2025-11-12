from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage
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
    user_question = state['messages'][-1].content
    thread_id = state.get("thread_id")

    # Build short history for context
    short_history = []
    for msg in state['messages'][-12:]:
        role = "User" if msg.type == "human" else "Assistant"
        short_history.append(f"{role}: {msg.content}")
    history_str = "\n".join(short_history) if short_history else "No prior messages."

    prompt = f'''
    You are a secure HR database assistant. 
    Use the thread_id in the WHERE clause to avoid data leakage.

    User query: {user_question}
    Thread ID (must use in SQL): {thread_id}

    --- Conversation so far ---
    {history_str}
    --- End ---

    Return only factual data from the DB. 
    If nothing found, say: "No data found."
    '''

    result = sql_agent_executor.invoke({"input": prompt})
    output = result.get("output", result)

    # CRITICAL: Return original messages + new state
    return {
        "sql_retrieval": output,
        "messages": state["messages"]
    }

    

def query(state: ChatState) -> ChatState:
    user_question = state["messages"][-1].content
    job_description = state.get("job_description", "")
    sql_retrieval = state.get("sql_retrieval", "No additional database info.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful HR assistant.

        Use this information to answer note that some or all information may not be relevant so answer honestly
        based on whatever information you have amd keep the answer short and concise:

        2. Job Description:
        {job_description}

        3. Information retreived from Database agent
        {sql_retrieval}
        """),
        ("human", "{user_question}")
    ])

    chain = prompt | chat_llm | StrOutputParser()
    response = chain.invoke({
        "job_description": job_description,
        "sql_retrieval": sql_retrieval,
        "user_question": user_question
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
db_conn = sqlite3.connect(f"{os.environ['CHAT_HISTORY_DIR']}/chat_history.db" , check_same_thread=False)
sqlite_memory = SqliteSaver(conn=db_conn)


workflow = graph.compile(checkpointer=sqlite_memory)