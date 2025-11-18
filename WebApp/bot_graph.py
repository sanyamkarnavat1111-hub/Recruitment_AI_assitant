# ==============================
# IMPORTS & GLOBAL SETUP
# ==============================
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from LLM_models import llm_query_rewritter , chat_llm 
from LLM_shcemas import ChatState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
import textwrap
from RAG import VectorStorage
from dotenv import load_dotenv
from custom_tool import SQLAgent


load_dotenv()

# OPT: Global vectorstore to avoid reloading embeddings on every call
vectorstore = VectorStorage()


# ==============================
# UTILITY: Format conversation history
# ==============================
def format_history(messages, max_messages=7):
    """Format last N messages as string for prompts."""
    short_msgs = messages[-max_messages:]
    history = []
    for msg in short_msgs:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        history.append(f"{role}: {msg.content}")
    return "\n".join(history)

# ==============================
# SQL RETRIEVER
# ==============================
def database_retriever(state: ChatState) -> ChatState:

    sql_agent = SQLAgent()

    thread_id = state["thread_id"]
    user_query = state['messages'][-1].content
    conversation_history = format_history(state['messages'], max_messages=7)

    
    
    try:
        sql_query = sql_agent.generate_sql_query(
        thread_id=thread_id,
        chat_history=conversation_history,
        user_query=user_query
        )
        # fixed_sql_query = sql_agent.sql_query_fixer(
        #     thread_id=thread_id,
        #     sql_query=sql_query
        # )

        output = sql_agent.execute_sql_query(sql_query=sql_query)
        return {"sql_retrieval": output}
    except Exception as e :
        return {"sql_retrieval": "Database retrieval failed..."}


# ==============================
# RAG RETRIEVER
# ==============================
def rag_retriever(state: ChatState) -> ChatState:

    user_query = state['messages'][-1].content
    thread_id = state["thread_id"]
    conversation_history = format_history(state['messages'], max_messages=7)

    try:

        prompt = ChatPromptTemplate.from_template(
            '''
            Understand the conversation history that will be shared to you and 
            rewrite the query replacing pronouns and other part of query with 
            knowledge that you got from understanding the conversation making the 
            retrieval of information from knowledge base more accurate if only required else keep the query as it is .
            For example :- User query can be like "What is her email",
            then understand the conversation history and replace her with relevant name
            if present in converstation history.

            # This is the conversation history :- 
            {conversation_history}

            # This is the query :- 
            {user_query} 
            '''
        )

        chain = prompt | llm_query_rewritter

        rewritten_query = chain.invoke(input={
            "conversation_history" : conversation_history,
            "user_query" : user_query
        }).rewritten_query

        docs = vectorstore.similarity_search(
                thread_id=thread_id,
                query=rewritten_query,
                top_k=3
            )
        if docs:
            return {"rag_retrieval": docs}
        else:
            {"rag_retrieval" : "No relevant documents founds "}
    except:
        try :
            docs = vectorstore.similarity_search(
                thread_id=thread_id,
                query=user_query,
                top_k=5
            )
            
            if docs:
                return {"rag_retrieval": docs}
            else:
                {"rag_retrieval" : "No relevant documents founds "}

        except Exception as e :
            return {"rag_retrieval" : f"rag retrieval failed {e}"}

# ==============================
# FINAL QUERY NODE
# ==============================
def query(state: ChatState) -> ChatState:

    user_query = state['messages'][-1]
    sql_retrieval = state["sql_retrieval"] if state["sql_retrieval"] else "No database info found."
    rag_retrieval = state["rag_retrieval"] if state["rag_retrieval"] else  "No document info found."
    # conversation_history = format_history(state['messages'], max_messages=7)


    prompt = ChatPromptTemplate.from_messages([
        ("system", textwrap.dedent("""
        You are a helpful HR assistant. You need to answer the query of users 
        without revealing sensitive information like thread-id and other irrelevant 
        information , just focus on what user has asked and based on the information
        that is provided to you encapsulate your answer to provide only 
        required details to answer query of the user. Also only provide the information
        do not reveal from where you got the data simply focus on providing the relevant answer that you 
        can answer of user. ALso check the data retrieved from database and RAG retrieval can overlap.
        Intelligently answer with sufficient information.
         
        Answer using only relevant info:

            Database retrieved data:
            {sql_retrieval}

            RAG retrieved data :
            {rag_retrieval}
         
        """)),
        ("human", "From the given information answer the query of the user {user_query}")
    ])

    chain = prompt | chat_llm | StrOutputParser()
    response = chain.invoke({
        "sql_retrieval": sql_retrieval,
        "rag_retrieval": rag_retrieval,
        # "conversation_history": conversation_history,
        "user_query" : user_query
    })

    return {"messages": [AIMessage(content=response)]}


# ==============================
# BUILD GRAPH
# ==============================
graph = StateGraph(ChatState)

graph.add_node("query", query)
graph.add_node("database_retriever", database_retriever)
graph.add_node("rag_retriever", rag_retriever)


graph.add_edge(START , "rag_retriever")
graph.add_edge(START , "database_retriever")
graph.add_edge("rag_retriever", "query")
graph.add_edge("database_retriever", "query")
graph.add_edge("query", END)

# ==============================
# CHECKPOINTER
# ==============================
os.makedirs(os.environ['CHAT_HISTORY_DIR'], exist_ok=True)
db_conn = sqlite3.connect(f"{os.environ['CHAT_HISTORY_DIR']}/chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn=db_conn)

workflow = graph.compile(checkpointer=memory)


# ==============================
# MAIN LOOP (TESTING)
# ==============================
if __name__ == "__main__":
    thread_id = "60306e06-822a-444a-a0c3-dc8bc488231f"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # --- INITIALIZE STATE ONLY IF NEEDED ---
    current_state = workflow.get_state(config)

    if current_state is None or not current_state.values.get("job_description"):
        print("Setting up new HR session with job description...")
        workflow.update_state(
            config=config,
            values={
                "thread_id": thread_id,
                "job_description": "", # Kept empty to check if sql agent can find job description on their own
                "messages": []
            }
        )
    else:
        print(f"Resuming existing session for thread: {thread_id}")

    # --- CHAT LOOP ---
    print("\nHR Chatbot Ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            result = workflow.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            print(f"AI: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"Error: {e}\n")