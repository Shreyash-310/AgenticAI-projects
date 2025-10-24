import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'D:\GenAI-Practice\AgenticAI-Projects\SmartAssistant\.env')
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_huggingface_embedding(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize and return a Hugging Face embedding model.
    Args:
        model_name: Name of the Hugging Face model to use.
    Returns:
        HuggingFaceEmbeddings instance or None if error occurs.
    """
    try:
        print(f"🔹 Initializing Hugging Face Embedding model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        print(f"⚠️ Error initializing Hugging Face Embedding model: {e}")
        return None

def get_gemini_embedding():
    """
    Initialize and return the Gemini Embedding model.
    Uses API key from environment variable GOOGLE_API_KEY.
    """
    try:
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        # if not api_key:
        #     raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            # google_api_key=api_key
        )
        return embeddings

    except Exception as e:
        print(f"⚠️ Error initializing Gemini Embedding model: {e}")
        return None