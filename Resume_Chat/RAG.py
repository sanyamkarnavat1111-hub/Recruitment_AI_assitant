from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import gemini_embeddings , llm_query_rewritter
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry , stop_after_attempt , wait_fixed
import os
from typing import List
from langchain_core.documents import Document

os.makedirs(os.environ['EMBEDDING_DIR'] , exist_ok=True)


class VectorStorage:

    def __init__(self):
        persist_directory = os.environ['EMBEDDING_DIR']

        self.vector_store = Chroma(
            embedding_function=gemini_embeddings,
            persist_directory=persist_directory
        )
    
    def store_general_embeddings(self, thread_id, documents):

        if not documents:
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splitted_documents = splitter.split_documents(documents)

        # Add metadata
        for doc in splitted_documents:
            doc.metadata["thread_id"] = thread_id
            doc.metadata["type"] = "general"

        self.vector_store.add_documents(splitted_documents)
        return splitted_documents


    def store_user_embeddings(self, thread_id: str, user_data: str):

        metadata = {
            "thread_id": thread_id,
            "type": "user"
        }

        # Store as a single text document
        self.vector_store.add_texts(
            texts=[user_data],       
            metadatas=[metadata]      
        )

        

    def similarity_search(
        self,
        thread_id: str,
        query: str,
        top_k: int = 3,
    ) -> List[Document]:
        """
        Always returns a list of Document objects.
        Empty list if nothing is found.
        """
        try:
            results: List[Document] = self.vector_store.similarity_search(
                query,
                k=top_k,
                filter={"thread_id": thread_id},
            )
            return results
        except Exception as e:
            print(f"[THREAD:{thread_id}] Vector store similarity_search failed: {e}")
            return []  # ← never return a string!
    
    @retry(stop=stop_after_attempt(5) , wait=wait_fixed(5))
    def rag_query_rewriter(self, user_query : str , conversation_history : str):
        try:

            query_rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at resolving pronouns, references, and ambiguous terms in user questions by using the conversation history.

            Your only job is to rewrite the user's latest question so that it becomes a standalone, highly effective search query for a vector database (RAG retrieval).

            Here are certain things that you can use as reference
            - Replace all pronouns (he, she, his, her, they, it, this candidate, that person, etc.) with the actual names, emails, job titles, or entities mentioned earlier.
            - If the user refers to "the last candidate", "the previous one", "the one from Bangalore", etc., replace it with the exact identifier from history.
            - Expand abbreviations if they were defined earlier (e.g., "JD" → "job description", "TCS" → "Tata Consultancy Services").
            - If the conversation mentions a specific resume, candidate, or job, include that context explicitly.
            - Add relevant filters only if they dramatically improve retrieval (e.g., location, years of experience, skill).
            - NEVER add information that is not present in the history.
            - If the original query is already clear and specific → return it unchanged.
            - Keep the rewritten query natural and concise (1–2 sentences max).
            - Output ONLY the rewritten query, nothing else. No explanations, no quotes, no markdown.
            """),

                ("human", """Conversation history:
            {conversation_history}

            Latest user question: {user_query}""")
            ])

            chain = query_rewriter_prompt | llm_query_rewritter

            rewritten_query = chain.invoke(input={
                "conversation_history" : conversation_history,
                "user_query" : user_query
            }).rewritten_query


            if rewritten_query:
                return rewritten_query
            else:
                return user_query
        except Exception as e :
            print(f"Error Re-writing user query ... :- {str(e)}")
            return user_query
