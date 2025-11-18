from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from LLM_models import gemini_embeddings
import os
from utils import parse_file


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

        

    def similarity_search(self, thread_id, query, top_k=3):

        results  = self.vector_store.similarity_search(
            query,
            k=top_k,
            filter={
                "thread_id": thread_id,
            }
        )

        if not results:
            return "No relevant results found."

        return results



if __name__ == "__main__":
    obj = VectorStorage()

    documents = parse_file(
        file_path="Uploads/data-scientist-resume-example.pdf",
        parsing_for_vector=True
    )

    
    res = obj.store_embeddings(
        thread_id="sam",
        documents=documents,
        type_of_doc="user"
    )

    print("Results :- \n" , res)


    


    


