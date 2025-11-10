from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from LLM_models import chat_llm
from LLM_shcemas import ChatState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

from dotenv import load_dotenv
load_dotenv()

CHAT_HISTORY_DIR = "Chat_history"
os.makedirs( CHAT_HISTORY_DIR, exist_ok=True)


def query(state: ChatState) -> ChatState:
    """Handle HR-related queries using persisted resume and job description data"""
    user_question = state["messages"][-1].content

    
    analyzed_resume_data = state.get("analyzed_resume_data", "")
    job_description = state.get("job_description", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful HR assistant. Answer to your fullest knowledge.

        Analyzed Resumes:
        {analyzed_resume_data}

        Job Description:
        {job_description}
        """),
        ("human", "{user_question}")
    ])

    chain = prompt | chat_llm | StrOutputParser()
    response = chain.invoke({
        "analyzed_resume_data": analyzed_resume_data,
        "job_description": job_description,
        "user_question": user_question
    })

    return {"messages": [AIMessage(content=response)]}


# Create the graph
graph = StateGraph(ChatState)
graph.add_node("query", query)
graph.add_edge(START, "query")
graph.add_edge("query", END)

# Sqlite 3 database checkpointer
db_conn = sqlite3.connect(f"{CHAT_HISTORY_DIR}/chat_history.db" , check_same_thread=False)
sqlite_memory = SqliteSaver(conn=db_conn)


workflow = graph.compile(checkpointer=sqlite_memory)