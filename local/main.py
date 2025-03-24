import os
from typing import List
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def populate_vector_db_secure(
    docs: List[Document],
    embedding_model: str = "text-embedding-ada-002"):
    """
    Embeds a list of Documents and stores them in a local FAISS vector index,
    using the OpenAI API key from the environment variable 'OPENAI_API_KEY'.
    
    :param docs: List of LangChain Document objects.
    :param embedding_model: The OpenAI embedding model name (e.g. 'text-embedding-ada-002').
    :param faiss_index_path: Directory path to save (or overwrite) the FAISS index.
    :return: A FAISS vector store object that contains the embeddings.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenAI API key found."
        )
    
    embedder = OpenAIEmbeddings(openai_api_key=api_key,
                                model=embedding_model)
    
    faiss_store = FAISS.from_documents(docs, embedder)
    faiss_store.save_local("vector-store")
    
    return faiss_store

