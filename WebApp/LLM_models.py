from langchain_groq import ChatGroq
from LLM_shcemas import ExtractResumeDataSchema


GROQ_API_KEY = "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"


chat_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

llm_resume_data_extractor = chat_llm.with_structured_output(schema=ExtractResumeDataSchema)
