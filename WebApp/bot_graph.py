from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from utils import parse_file, analyze_resume, classify_query
from LLM_models import chat_llm
from LLM_shcemas import ChatState
from typing import Literal

from dotenv import load_dotenv
import os

load_dotenv()

UPLOADS = "Uploads"




def query(state: ChatState) -> ChatState:
    """Handle HR-related queries using resume and job description data"""
    user_question = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful HR assistant.
        Summary of the analysis of resume:
        {analyzed_resume_data}

        Job description:
        {job_description}

        Resume data:
        {resume_data}
        """),
        ("human", "{user_question}")
    ])

    chain = prompt | chat_llm | StrOutputParser()
    response = chain.invoke({
        "analyzed_resume_data": state["analyzed_resume_data"],
        "job_description": state["job_description"],
        "resume_data": state["resume_data"],
        "user_question": user_question
    })

    return {"messages": [AIMessage(content=response)]}


# Create the graph
graph = StateGraph(ChatState)

# Add only the actual processing nodes (NOT the routing function)
graph.add_node("query", query)

graph.add_edge(START , "query")
graph.add_edge("query", END)

# Compile with memory
memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)