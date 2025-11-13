from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from LLM_models import gemini_embeddings
import os


os.makedirs(os.environ['EMBEDDING_DIR'] , exist_ok=True)


class VectorStorage:

    def __init__(self):
        persist_directory = os.environ['EMBEDDING_DIR']

        self.vector_store = Chroma(
            embedding_function=gemini_embeddings,
            persist_directory=persist_directory  
        )

        
    
    def store_embeddings(self, thread_id, documents):

        # Parse the file and split into documents
        if not documents:
            return None

        # Split documents into chunks for embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_documents = splitter.split_documents(documents=documents)

        # Add the thread_id as metadata to each document
        # We modify the documents to include thread_id
        documents_with_metadata = []
        for doc in splitted_documents:
            doc.metadata['thread_id'] = thread_id
            documents_with_metadata.append(doc)

        # Add the documents with the additional metadata to Chroma
        self.vector_store.add_documents(documents=documents_with_metadata)


    def similarity_search(self, thread_id , query , top_k=3):

        results = self.vector_store.similarity_search(
            query, k=top_k, filter={"thread_id": thread_id}
        )
        if results:
            retrieved_data = "Retrieved data from knowledge base .\n"
            for res in results:
                retrieved_data += "-"*20
                retrieved_data += f"{res.page_content} [{res.metadata}]"
            return results

        return "No relevant results found from document"

if __name__ == "__main__":
    obj = VectorStorage()
    res = obj.convert_to_embeddings(file_path="Uploads/data-scientist-resume-example.pdf" , thread_id="sam")

    


    


