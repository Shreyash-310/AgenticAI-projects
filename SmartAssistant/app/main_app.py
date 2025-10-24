"""import streamlit as st
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
"""

import streamlit as st
from graph import run_chatbot
from core.document_ingestion import process_uploaded_files

st.set_page_config(page_title="Smart Assistant", layout="wide")

st.title("🤖 Smart AI Assistant with LangGraph + Gemini")
st.caption("Interact with documents, databases, and general knowledge in one unified chat.")

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Sidebar for uploads ---
st.sidebar.header("📄 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.sidebar.button("Process Files"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            # Use your existing ingestion function
            process_uploaded_files(uploaded_files)
            st.session_state.uploaded_files = [f.name for f in uploaded_files]
        st.sidebar.success("✅ Documents processed successfully!")
    else:
        st.sidebar.warning("Please upload at least one document.")

# --- Chat Interface ---
st.header("💬 Chat with Assistant")

user_query = st.text_input("Ask your question here:")
if st.button("Send") and user_query.strip():
    with st.spinner("Thinking..."):
        state = run_chatbot(user_query)
    
    # Store in session
    st.session_state.chat_history.append(
        {"query": user_query, "response": state.answer, "trace": state.reasoning_trace}
    )

# --- Chat History ---
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(chat["response"])
        if chat["trace"]:
            with st.expander("🧠 View reasoning trace"):
                st.markdown(chat["trace"])

# --- Sidebar Information ---
st.sidebar.divider()
st.sidebar.markdown("### 🧩 Active Files")
if st.session_state.uploaded_files:
    for f in st.session_state.uploaded_files:
        st.sidebar.markdown(f"- {f}")
else:
    st.sidebar.markdown("_No documents uploaded yet._")
