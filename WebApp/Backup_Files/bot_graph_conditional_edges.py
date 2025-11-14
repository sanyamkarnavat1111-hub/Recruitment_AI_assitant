# ==============================
# IMPORTS & GLOBAL SETUP
# ==============================
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from LLM_models import chat_llm, llm_router
from LLM_shcemas import ChatState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from sql_agent import sql_agent_executor
from tenacity import retry, wait_fixed, stop_after_attempt
from typing import Literal
import os
import textwrap
from RAG import VectorStorage
from dotenv import load_dotenv
from utils import remove_extra_space

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
# ROUTER: Decide rag / sql / none
# ==============================
def decision_router(state: ChatState) -> Literal['rag', 'sql', 'none']:

    if state['messages']:

        user_msg = state['messages'][-1]
        user_query = user_msg.content
        conversation_history = format_history(state['messages'], max_messages=7)
    else:
        return 'none'
    job_description = state.get('job_description', '')  # FIX: Safe access

    prompt = PromptTemplate(
        template=textwrap.dedent('''
            You are a smart HR query router. Decide if the answer can be given from:
            - 'rag': Need to retrieve from uploaded documents (e.g. company policy, handbook)
            - 'sql': Need candidate data from database (name, score, shortlist,email,work experience details etc.)
            - 'none': Answer is already in chat history or general knowledge

            Chat History:
            {chat_history}

            User Query: {user_question}

            Job Description: {job_description}

            Return ONLY one word: rag, sql, or none.
            '''),
        input_variables=['chat_history', 'user_question', 'job_description']
    )

    chain = prompt | llm_router

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def get_valid_decision():
        response = chain.invoke({
            "chat_history": conversation_history,
            "user_question": user_query,
            "job_description": job_description
        })
        decision = response.decision
        if decision not in ('rag', 'sql', 'none'):
            raise ValueError(f"Invalid decision: {decision}")
        return decision

    try:
        decision = get_valid_decision()
        print("Decision :-" , decision)

        return decision
    except Exception:
        return 'none'


# ==============================
# SQL RETRIEVER
# ==============================
def database_retriever(state: ChatState) -> ChatState:
    thread_id = state["thread_id"]
    user_query = state['messages'][-1].content
    job_description = state.get("job_description", "")
    conversation_history = format_history(state['messages'], max_messages=7)

    prompt = textwrap.dedent(f'''
        You are a secure HR SQL agent.
        ALWAYS use `thread_id = '{thread_id}'` in WHERE clause.
        Return only factual data. If none, say: "No data found."

        User Query: {user_query}
        Job Description: {job_description}

        Conversation History:
        {conversation_history}

        Generate SQL and return result.
        ''')

    result = sql_agent_executor.invoke({"input": prompt})
    output = result.get("output", "No data found.")

    return {"sql_retrieval": output}


# ==============================
# RAG RETRIEVER
# ==============================
def rag_retriever(state: ChatState) -> ChatState:
    user_query = state['messages'][-1].content
    thread_id = state["thread_id"]

    docs = vectorstore.similarity_search(
        thread_id=thread_id,
        query=user_query,
        top_k=5
    )

    return {"rag_retrieval": docs}


# ==============================
# FINAL QUERY NODE
# ==============================
def query(state: ChatState) -> ChatState:
    job_description = state.get("job_description", "")
    sql_retrieval = state.get("sql_retrieval", "No database info.")
    rag_retrieval = state.get("rag_retrieval", "No document info.")
    conversation_history = format_history(state['messages'], max_messages=7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant. Be concise and honest."),
        ("human", textwrap.dedent('''
            Answer using only relevant info:

            DB Data:
            {sql_retrieval}

            Document Info:
            {rag_retrieval}

            Chat History:
            {conversation_history}

            Job Description:
            {job_description}

            Keep answer short and clear.
            '''))
    ])

    chain = prompt | chat_llm | StrOutputParser()
    response = chain.invoke({
        "sql_retrieval": sql_retrieval,
        "rag_retrieval": rag_retrieval,
        "conversation_history": conversation_history,
        "job_description": job_description
    })

    return {"messages": [AIMessage(content=response)]}


# ==============================
# BUILD GRAPH
# ==============================
graph = StateGraph(ChatState)

graph.add_node("query", query)
graph.add_node("database_retriever", database_retriever)
graph.add_node("rag_retriever", rag_retriever)

# Conditional routing from START
graph.add_conditional_edges(
    START,
    decision_router,
    {
        "rag": "rag_retriever",
        "sql": "database_retriever",
        "none": "query"
    }
)

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
    thread_id = "sanyam"
    job_description = remove_extra_space(text='''
    Job Title: Data Scientist

    Location:

    Hybrid – Bengaluru, India (3 days in office / 2 days remote)

    Experience Required:

    Minimum 1 years of professional experience in data science, analytics, or machine learning

    Employment Type:

    Full-time



    About the Role

    We are seeking a Data Scientist with strong analytical and technical skills to join our growing Data & AI team. The ideal candidate will be responsible for designing, developing, and deploying data-driven solutions that power key business decisions and product innovations. You will collaborate with cross-functional teams to identify opportunities for leveraging data and machine learning to drive impact.



    Key Responsibilities

    Collect, clean, and preprocess large datasets from multiple sources

    Build predictive and classification models using machine learning and statistical techniques

    Analyze data trends and provide actionable business insights to stakeholders

    Develop and maintain data pipelines and ETL processes in collaboration with data engineering teams

    Evaluate model performance, tune hyperparameters, and ensure robustness in production environments

    Communicate findings through visualizations and clear, data-driven storytelling

    Stay up-to-date with the latest developments in AI/ML and data science tools



    Required Qualifications

    Bachelor’s or Master’s degree in Computer Science, Statistics, Mathematics, Data Science, or related field

    3 years of hands-on experience in data science or applied machine learning

    Proficiency in Python (pandas, NumPy, scikit-learn, matplotlib, seaborn.

    Strong experience with SQL and data querying from relational databases

    Practical understanding of supervised and unsupervised learning algorithms

    Experience with data visualization tools such as Power BI, Tableau, or Plotly Dash

    Familiarity with cloud platforms (AWS, GCP, or Azure)

    Excellent problem-solving skills and business acumen



    Preferred Qualifications

    Experience deploying models using FastAPI, Flask, or MLflow

    Familiarity with deep learning frameworks (TensorFlow, PyTorch)

    Exposure to big data tools (Spark, Databricks, or Hadoop ecosystem)

    Understanding of MLOps or CI/CD for model deployment and monitoring

    Prior experience in domains such as fintech, e-commerce, or marketing analytics



    Soft Skills

    Strong communication and collaboration skills

    Ability to translate complex analyses into clear business recommendations

    Self-motivated with a growth mindset and attention to detail
    ''')
    config = {"configurable": {"thread_id": thread_id}}
    
    # --- INITIALIZE STATE ONLY IF NEEDED ---
    current_state = workflow.get_state(config)

    if current_state is None or not current_state.values.get("job_description"):
        print("Setting up new HR session with job description...")
        workflow.update_state(
            config=config,
            values={
                "thread_id": thread_id,
                "job_description": job_description,
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