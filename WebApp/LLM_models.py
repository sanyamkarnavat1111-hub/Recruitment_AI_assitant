from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from LLM_shcemas import ExtractResumeDataSchema , ClassifyQuery , ResumeAnalysis
from dotenv import load_dotenv
import os


load_dotenv()



GROQ_API_KEY = os.environ['GROQ_API_KEY']

chat_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# chat_llm = ChatOllama(
#     model="llama3:8b",
# )

llm_resume_data_extractor = chat_llm.with_structured_output(schema=ExtractResumeDataSchema)
llm_sentiment_analyzer = chat_llm.with_structured_output(schema=ClassifyQuery)
llm_resume_analysis = chat_llm.with_structured_output(schema=ResumeAnalysis)

