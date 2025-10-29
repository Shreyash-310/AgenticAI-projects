import streamlit as st
import os
from core.ingestion import load_and_split_documents
from core.embedding_store import create_faiss_index
from langchain.schema import AIMessage, HumanMessage
from graph import *
from chatbot.state import ChatbotState

def app_session_init():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage("Hello, How can I help you?")]

    chat_history = st.session_state["chat_history"]

    for history in chat_history:
        if isinstance(history, AIMessage):
            st.chat_message("ai").write(history.content)
        if isinstance(history, HumanMessage):
            st.chat_message("human").write(history.content)

def run():

    st.set_page_config(page_title="Smart Assistant", layout="wide")
    st.title("📘 Smart Assistant Chatbot")

    UPLOAD_DIR = "data/pdf_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    DB_DIR = "data/db_uploads"
    os.makedirs(DB_DIR, exist_ok=True)

    if 'is_ready' not in st.session_state:
        st.session_state.is_ready = False
        st.session_state.app = None
        st.session_state.query_result = None  # Store query results

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    pdf_files = st.sidebar.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True)

    db_file = st.sidebar.file_uploader(
        "Upload DataBase File",
        type = ["db", "sqlite"])

    process_btn = st.sidebar.button('Process Files')

    if process_btn:
        if not pdf_files or not db_file:
            st.sidebar.warning("Please upload at least one files before processing...")
        else:
            with st.spinner("Processing uploaded files ..."):
                saved_paths = []
                for file in pdf_files:
                    file_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_paths.append(file_path)

                chunks = load_and_split_documents(saved_paths)
                _ = create_faiss_index(chunks)
            
            with st.spinner("Processing database file ..."):
                saved_db_path = None
                if db_file:
                    db_path = os.path.join(DB_DIR, db_file.name)
                    with open(db_path, "wb") as f:
                        f.write(db_file.getbuffer())
                    saved_db_path = db_path

            st.sidebar.success("Documents processed successfully !!!")
            st.session_state.is_ready = True
            try:
                st.session_state.initial_state = ChatbotState()
                st.session_state.app = build_chatbot_graph()
            except Exception as e:
                print(f"Error occured as {e}")

    for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.write(message['content'])


    if user_query:= st.chat_input("Enter your query here ..."):
        if not st.session_state.is_ready:
            st.info("Please uplaod document for chatting ...")
        else:
            st.session_state.messages.append({"role":"user","content":user_query})
            with st.chat_message("user"):
                st.write(user_query)
            try:
                with st.chat_message('assistant'):
                    st.session_state.query_result = st.session_state.app.invoke(
                                    {'query':user_query})
                    print(f"result => {st.session_state.query_result}")
                    print(f"result => {st.session_state.query_result['answer']}")
                    # Add to history
                    st.session_state.messages.append({
                        'role':'assistant',
                        'content':st.session_state.query_result['answer']
                    })
                # with st.chat_message("assistant"):
                    st.write(st.session_state.query_result['answer'])
            except Exception as e:
                st.error(f"Error: {str(e)}")                


if __name__ == "__main__":
    run()
