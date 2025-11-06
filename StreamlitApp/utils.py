import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document


def convert_to_documents(file_path: str) -> list[Document]:
    """
    Load a document (PDF, DOCX, or TXT) into LangChain Document objects.
    Automatically reads all pages for PDFs and returns a list of Documents.
    """
    # Ensure file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get lowercase extension (e.g., '.pdf')
    extension = Path(file_path).suffix.lower()

    # Select appropriate loader
    if extension == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    # Load all pages/content
    documents = loader.load()

    # Validate non-empty result
    if not documents:
        raise ValueError(f"No content loaded from file: {file_path}")

    return documents
