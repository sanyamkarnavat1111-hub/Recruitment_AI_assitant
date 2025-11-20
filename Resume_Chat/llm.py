from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from schemas import (
    ExtractResumeDataSchema , 
    DocumentClassifier , 
    CreateSqlQuery , 
    FixSqlQuery,
    RagQueryRewritter
)


import os
from dotenv import load_dotenv



load_dotenv()


groq_llm_1 = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ['GROQ_API_KEY_1'],
)


groq_llm_2 =  ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ['GROQ_API_KEY_2'],
)



chat_llm_ollama = ChatOllama(
    model="llama3:8b",
    num_gpu=1,
    temperature=0.8
)

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001" , google_api_key=os.environ['GOOGLE_API_KEY'])


llm_resume_data_extractor = groq_llm_2.with_structured_output(schema=ExtractResumeDataSchema)
llm_document_classifier = groq_llm_1.with_structured_output(schema=DocumentClassifier)
llm_sql_query_generator = chat_llm_ollama.with_structured_output(schema=CreateSqlQuery)
llm_sql_fixer = chat_llm_ollama.with_structured_output(schema=FixSqlQuery)
llm_query_rewritter = groq_llm_2.with_structured_output(schema=RagQueryRewritter)
