from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from LLM_shcemas import (
    ExtractResumeDataSchema , 
    ResumeAnalysis , 
    QueryRouter , 
    CreateSqlQuery , 
    FixSqlQuery,
    RagQueryRewritter
)
from dotenv import load_dotenv
import os


load_dotenv()

# chat_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key = os.environ['GOOGLE_API_KEY']
# )



chat_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ['GROQ_API_KEY_1'],
    temperature=0.7,
    streaming=True
)

chat_llm_2  = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ['GROQ_API_KEY_2'],
    temperature=0.8,
    streaming=True
)



chat_llm_ollama = ChatOllama(
    model="llama3:8b",
    num_gpu=1,
    temperature=0.8
)

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001" , google_api_key=os.environ['GOOGLE_API_KEY'])
ollama_embeddings = OllamaEmbeddings(model="llama3:8b")

llm_resume_data_extractor = chat_llm_2.with_structured_output(schema=ExtractResumeDataSchema)
llm_resume_analysis = chat_llm.with_structured_output(schema=ResumeAnalysis)
llm_router = chat_llm_ollama.with_structured_output(schema=QueryRouter)
llm_sql_query_generator = chat_llm_ollama.with_structured_output(schema=CreateSqlQuery)
llm_sql_fixer = chat_llm_ollama.with_structured_output(schema=FixSqlQuery)
llm_query_rewritter = chat_llm.with_structured_output(schema=RagQueryRewritter)