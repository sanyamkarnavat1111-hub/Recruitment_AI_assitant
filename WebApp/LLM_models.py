from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from LLM_shcemas import ExtractResumeDataSchema , ResumeAnalysis
from dotenv import load_dotenv
import os


load_dotenv()

chat_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key = os.environ['GOOGLE_API_KEY']
)

# chat_llm = ChatOllama(
#     model="llama3:8b",
# )

chat_llm_groq = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ['GROQ_API_KEY'],
    temperature=0.7,
    streaming=True
)

llm_resume_data_extractor = chat_llm_groq.with_structured_output(schema=ExtractResumeDataSchema)
llm_resume_analysis = chat_llm_groq.with_structured_output(schema=ResumeAnalysis)

