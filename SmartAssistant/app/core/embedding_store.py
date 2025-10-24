import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from .embedding_util import get_gemini_embedding, get_huggingface_embedding
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv(dotenv_path='D:\GenAI-Practice\AgenticAI-Projects\SmartAssistant\.env')

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def create_faiss_index(documents, index_path="data/faiss_index", 
                       model="models/embedding-001"):
    """
    Generate embeddings for documents using Gemini and store in FAISS.

    Args:
        documents (list): LangChain Document chunks.
        index_path (str): Path to save FAISS index.
        model (str): Gemini embedding model name.

    Returns:
        FAISS: Vector store object.
    """
    try:
        os.makedirs(index_path, exist_ok=True)

        embeddings = get_huggingface_embedding() #get_gemini_embedding()      
        print("Huggging Face Model Loaded Successfully.....")  
        if embeddings is None:
            raise RuntimeError("Embedding model could not be initialized.")
        input('Before creating faiss vector store')
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(index_path)
        print(f"✅ FAISS index created and saved at: {index_path}")
        return vectorstore

    except Exception as e:
        print(f"⚠️ Error creating FAISS vector store: {e}")
        return None
    