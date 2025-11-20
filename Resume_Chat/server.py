from fastapi import FastAPI , Depends , Form , UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
import tempfile
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from utils import (
    parse_file , 
    remove_extra_space , 
    classify_document , 
    extract_data_from_resume,
)
from database import (
    create_table , 
    test_connection , 
    drop_table,
    insert_extracted_data,
    insert_job_description
)
from schemas import ChatRequest
from RAG import VectorStorage
from llm import (
    groq_llm_1
)
from Database_Agent import SQLAgent


import os
import uvicorn
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResumeAI")


vectorStore = VectorStorage()
sql_agent = SQLAgent()

# Dependency function to check thread_id
def check_thread_id(thread_id: str = Form(...)) -> str:
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="Thread ID is required.")
    return thread_id


@app.post("/upload_files")
async def upload_files(
    file : UploadFile,
    thread_id :str = Depends(check_thread_id),
):
    try :
        temp_file_path : str | None = None
        if not file.filename:
            raise HTTPException(status_code=400 , detail="File is required")

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in {"pdf", "txt", "docx"}:
            raise HTTPException(status_code=400, detail=f"Unsupported JD file type: .{ext}")

        raw_bytes = await file.read()
        if not raw_bytes:
            raise ValueError("Job description file is empty.")


        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        temp_file.write(raw_bytes)
        temp_file.close()
        temp_file_path = temp_file.name

        logger.info(f"[THREAD {thread_id}] Temp file created: {temp_file_path}")

        file_text = parse_file(file_path=temp_file_path)
        file_text = remove_extra_space(text=file_text)
        
        # Write the code to check if this is general document or resume file extract resume data and tore
        document_type = classify_document(text=file_text)
        if document_type == "resume":
            logger.info(f"[THREAD {thread_id}] Document type is {document_type}...")
            extracted_resume_data = extract_data_from_resume(resume_data=file_text)

            candidate_name = str(extracted_resume_data['candidate_name']).lower()
            formatted_user_data = f'''
            Candidate name : {candidate_name}
            Job role of {candidate_name} : {extracted_resume_data['job_role']}

            Location of {candidate_name} : {extracted_resume_data['location']}

            Contact information - info
                - mail of {candidate_name} : {extracted_resume_data['email_address']}
                - Contact number - Phone number of {candidate_name} : {extracted_resume_data['contact_number']}
                - Linked In of {candidate_name} :- {extracted_resume_data['linkedin_url']} 
            Total experience of {candidate_name} :- {extracted_resume_data['total_experience']}

            '''
            # Add thread id to the extracted resume data
            extracted_resume_data.update({
                "thread_id" : thread_id
            })
            # Store the data in database
            insert_extracted_data(
                extracted_resume_data=extracted_resume_data
            )
            logger.info(f"[THREAD {thread_id}] Inserted data to database ...")
            vectorStore.store_user_embeddings(
                thread_id=thread_id,
                user_data=formatted_user_data
            )
            logger.info(f"[THREAD {thread_id}] Inserted data to Vector store  ...")
        elif document_type =="job_description":
            logger.info(f"[THREAD {thread_id}] Document type is :- {document_type}...")
            # Create vector embedding
            logger.info(f"[THREAD {thread_id}] Parsing document ...")
            documents = parse_file(file_path=temp_file_path , parsing_for_vector=True)
            insert_job_description(
                thread_id=thread_id,
                job_description=file_text # Using the same file text which was extracted earlier so that we do not need to parse the file again.
            )
            logger.info(f"[THREAD {thread_id}] Inserted document to database ...")
            # Storing job description as user embedding since there is not point of splitting the entire job description 
            vectorStore.store_user_embeddings(
                    thread_id=thread_id,
                    user_data=file_text
                )
            logger.info(f"[THREAD {thread_id}] Storing data to vector store ...")
        else:
            logger.info(f"[THREAD {thread_id}] Document type is :- {document_type}...")
            documents = parse_file(file_path=temp_file_path , parsing_for_vector=True)

            vectorStore.store_general_embeddings(
                thread_id=thread_id,
                documents=documents
            )
        logger.info(f"[THREAD {thread_id}] File parsed successfully ...")
        return JSONResponse({
            "success" : True,
            "message" : "File upload successfull ..."
        })
    except Exception as e :
        logger.error(f"[THREAD:{thread_id}] Error uploading files :- {str(e)}")

        return JSONResponse({
            "success" : True,
            "message" : f"Error uploading file :- {e}"
        })

@app.post("/chat")
async def chat(request : ChatRequest):
    try:
        if not request.thread_id:
            raise HTTPException(status_code=400 , detail="Thread id missing ...")
        logger.info(f"[THREAD:{request.thread_id}] Retireving history ..")
        history_store = SQLChatMessageHistory(
            session_id=request.thread_id,
            connection=f"sqlite:///{os.environ['CHAT_HISTORY_DIR']}/chat_history.db"
        )

        # Update the history for Human message
        history_store.add_user_message(request.user_query)

        full_history = history_store.messages  # List[BaseMessage]
        short_history = full_history[-6:] if len(full_history) > 6 else full_history
        history_str = "\n".join([f"Human: {m.content}\nAI: {m.content}" if isinstance(m, AIMessage) else f"Human: {m.content}" for m in short_history])
        
        try:
            logger.info(f"[THREAD:{request.thread_id}] generating SQL query ..")
            # Write code for RAG retireval and database retrieval
            sql_query = sql_agent.generate_sql_query(
                thread_id=request.thread_id,
                chat_history=history_str,
                user_query=request.user_query
            )

            logger.info(f"[THREAD:{request.thread_id}] Executing query :- {sql_query}..")

            
            database_output = sql_agent.execute_sql_query(sql_query=sql_query)

            if database_output:
                logger.info(f"[THREAD:{request.thread_id}] Information retrieval from database has some data ..")
            else:
                database_output = "No relevant information found in database .."

                logger.info(f"[THREAD:{request.thread_id}] Not results found from database ..")

        except Exception as e :
            logger.error(f"[THREAD:{request.thread_id}] Error retrieving information from databaset :- {str(e)}")

        rag_str = ""
        try:
            logger.info(f"[THREAD:{request.thread_id}] Retrieving information via RAG ..")
            
            rewritten_query = vectorStore.rag_query_rewriter(
                user_query=request.user_query,
                conversation_history=short_history
            )

            rag_output= vectorStore.similarity_search(
                thread_id=request.thread_id,
                query=rewritten_query
            )
            logger.info(f"[THREAD:{request.thread_id}] RAG retrieval has some data ..")
            if rag_output:
                rag_str = "\n".join([doc.page_content for doc in rag_output])  

            else:
                rag_str = "No relevant documents found from RAG"
                logger.info(f"[THREAD:{request.thread_id}] No relevant info found from RAG ..")

        except Exception as e :
            logger.error(f"[THREAD:{request.thread_id}] Error retrieving info from RAG ...")


        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful Human recruitment assistant. Your job is to study the information provided to you and answer the user queries to your fullest knowledge."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}\n\nDatabase Results:\n{database_output}\n\nRAG Results:\n{rag_str}"),  # Fixed placeholder
        ])

        chain = prompt | groq_llm_1 | StrOutputParser()
        logger.info(f"[THREAD:{request.thread_id}] Creating chain ...")
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=lambda thread_id: SQLChatMessageHistory(
                session_id=thread_id,
                connection=f"sqlite:///{os.environ['CHAT_HISTORY_DIR']}/chat_history"
            ),
            input_messages_key="input",  
            history_messages_key="history"
        )
        response = chain_with_history.invoke(
            {
                "input": request.user_query,
                "database_output": str(database_output),
                "rag_str": rag_str
            },
            config={"configurable": {"session_id": request.thread_id}}
        )
        # Update the history for AI 
        history_store.add_ai_message(response)

        return JSONResponse({"success": True, "message": response}) 
    except Exception as e :
        return JSONResponse({
            "success" : True,
            "message" : f"Error uploading file :- {e}"
        })



if __name__ == "__main__":
    
    logger.info(f"GROQ API key 1:- {os.environ['GROQ_API_KEY_1']}")
    logger.info(f"GROQ API key 1:- {os.environ['GROQ_API_KEY_2']}")
    
    os.makedirs(os.environ.get('DATABASE_DIR', ''), exist_ok=True)
    os.makedirs(os.environ.get('CHAT_HISTORY_DIR', ''), exist_ok=True)
    os.makedirs(os.environ.get("EMBEDDING_DIR",'') , exist_ok=True)

    test_connection()
    drop_table()
    create_table()


    port = 3333
    logger.info(f"Starting server on port {port}")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )