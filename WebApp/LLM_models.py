from langchain_groq import ChatGroq
from LLM_shcemas import ExtractResumeDataSchema , ClassifyQuery
from dotenv import load_dotenv
import os


load_dotenv()



GROQ_API_KEY = os.environ['GROQ_API_KEY']

chat_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

llm_resume_data_extractor = chat_llm.with_structured_output(schema=ExtractResumeDataSchema)
llm_sentiment_analyzer = chat_llm.with_structured_output(schema=ClassifyQuery)
