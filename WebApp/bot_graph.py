from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from utils import parse_file , analyze_resume
from LLM_models import chat_llm 

from dotenv import load_dotenv
import os

load_dotenv()


UPLOADS = "Uploads"






# === State Definition ===
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_thread: str
    analyzed_resume_data : str
    job_description : str
    resume_data : str






# === Node: Answer User Query Using Analyzed Data ===
def query(state: ChatState) -> ChatState:
    messages = state["messages"]
    user_question = messages[-1].content

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful HR assistant...
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



# === Build Graph ===
graph = StateGraph(ChatState)
graph.add_node("query", query)

graph.add_edge(START , "query")
graph.add_edge("query", END)

# Compile with persistent memory
memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)



if __name__ == "__main__":
    print("=== Resume-Aware Chat with Thread Persistence ===")
    print("Commands: 'exit', 'new thread'\n")

    thread_id = "sanyam"

    # === Load Resume & Job Description from Files ===
    resume_path = os.path.join(UPLOADS, "flutter_resume.pdf")
    job_path = os.path.join(UPLOADS, "job_description.docx")

    resume_data = parse_file(resume_path) 
    job_description = parse_file(job_path)

    resume_analysis = analyze_resume(
        resume_data=resume_data,
        job_description=job_description,
    )

    if not resume_data:
        print(f"Warning: No resume found at {resume_path}")
    if not job_description:
        print(f"Warning: No job description found at {job_path}")
    

    print(f"\n[{thread_id}] AI Analysis:\n{ resume_analysis}\n")

    while True:

        # === User Input ===
        user_input = input(f"[{thread_id}] You: ").strip()
        if user_input.lower() == 'exit':
            break


        config = {"configurable": {"thread_id": thread_id}}

        # === Build Input State ===
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "conversation_thread": thread_id,
            "analyzed_resume_data" : resume_analysis,
            "job_description" : job_description,
            "resume_data" : resume_data
        }

        try:
            # === Run Workflow (No Streaming) ===
            result = workflow.invoke(input_state, config=config)
            ai_message = result['messages'][-1]
            
            print(f"\n[{thread_id}] AI :\n{ ai_message.content}\n")

        except Exception as e:
            print(f"Error: {e}")