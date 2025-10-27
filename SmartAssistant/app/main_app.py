import streamlit as st
import os
from core.ingestion import load_and_split_documents
from core.embedding_store import create_faiss_index

st.set_page_config(page_title="Smart Assistant - Document Indexing", layout="wide")
st.title("📘 Smart Assistant - Document Ingestion and Embedding")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Documents"):
    saved_paths = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)

    st.info("Processing uploaded files...")
    chunks = load_and_split_documents(saved_paths)
    st.success(f"Loaded and chunked {len(chunks)} sections.")

    st.info("Creating FAISS embeddings using Gemini...")
    create_faiss_index(chunks)
    st.success("✅ Embeddings created and saved to FAISS index!")

    st.write("You can now proceed to build your **retrieval & chat** interface.")
 
