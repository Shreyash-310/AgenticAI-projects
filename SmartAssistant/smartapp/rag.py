from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
# from chatbot.state import ChatbotState
from chatbot.state import ChatbotState

import os
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv(dotenv_path='D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/.env')

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

def rag_node(state: ChatbotState) -> ChatbotState:
    """
    Handles document-based queries using FAISS + LLM.
    Retrieves relevant chunks and generates an answer.
    """
    try:
        # Step 1: Load embeddings and FAISS vector store
        embeddings = get_huggingface_embedding()
        vectorstore = FAISS.load_local("D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/app/data/faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Step 2: Create LLM (Gemini or fallback)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

        # Step 3: Create Retrieval-QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Step 4: Run the query
        result = qa_chain.invoke({"query": state.query})

        # Step 5: Update state
        state.answer = result["result"]
        state.context = [doc.page_content for doc in result["source_documents"]]
        state.reasoning_trace = (
            "Used FAISS semantic search to find relevant chunks, then used Gemini for synthesis."
        )
        return state

    except Exception as e:
        state.answer = f"RAG failed: {e}"
        state.reasoning_trace = "RAG node failed to process query."
        return state

if __name__ == '__main__':

    state = ChatbotState(query="Summarize the key points from the uploaded document.", query_type="rag")
    updated = rag_node(state)
    print(f"Answer:\n{updated.answer}\n")
    print(f"Context used:\n{updated.context[:1]}\n")
    print(f"Reasoning: {updated.reasoning_trace}")