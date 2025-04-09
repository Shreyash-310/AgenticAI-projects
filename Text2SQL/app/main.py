import streamlit as st

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text, inspect
import sqlparse
from langchain.schema import AIMessage, HumanMessage

from nlp2sql import *

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

    st.title("NLP to SQL Workflow")
    
    app_session_init()
    
    # Step 1: Get the database path (once per session)
    if "db_path" not in st.session_state:
        st.session_state.db_path = None
        st.session_state.workflow_generated = False
        st.session_state.show_image = False  # Image visibility state
        st.session_state.query_result = None  # Store query results
        st.session_state.app = None 
        st.session_state.workflow_image = None

    st.sidebar.header("Settings")

    image_path = "graph_image.png"

    # Input for database path
    db_path = st.sidebar.text_input("Enter Database Absolute Path:", value=st.session_state.db_path or "")

    # Save the path in session state
    if db_path:
        st.session_state.db_path = db_path
        try:
            engine = create_engine(f"sqlite:///{db_path}")
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            Base = declarative_base()
            db_schema = get_database_schema(engine)
        except Exception as e:
            st.sidebar.error(f"Error connecting to database: {e}")

    if db_path and not st.session_state.workflow_generated:
        try:
            st.session_state.app = create_workflow(schema=db_schema, session=SessionLocal)
            
            image_path = "graph_image.png"
            img_data = st.session_state.app.get_graph(xray=True).draw_mermaid_png()
            with open(image_path, "wb") as f:
                f.write(img_data)
            st.session_state.workflow_generated = True

        except Exception as e:
            st.sidebar.error(f"Error generating workflow: {e}")

    def toggle_image():
        st.session_state.show_image = not st.session_state.show_image

    st.sidebar.button("Show/Hide Workflow", on_click=toggle_image)

    # query = st.text_input("Enter your SQL Query:")
    query = st.chat_input("Add your prompt...")

    if st.session_state.show_image and os.path.exists(image_path):
        st.image(image_path, caption="Generated Workflow", use_container_width=True)

    if query and query != st.session_state.get("last_query"):
        if st.session_state.app:
            st.chat_message('user').write(query)
            st.session_state["chat_history"] += [HumanMessage(query)]
            try:
                with st.spinner("Processing....."):
                    st.session_state.query_result = st.session_state.app.invoke(
                        {"question": query, "attempts": 0},
                        config={"configurable": {"current_user_id": "1"}}
                        )
                st.session_state.last_query = query  # Store last processed query
            except Exception as e:
                st.error(f"Error executing query: {e}")
                st.session_state.query_result = {"sql_query": "N/A", "query_result": "Error executing query"}
        else:
            st.error("Workflow has not been generated. Please check database path.")

    # Step 5: Display Query Results (Without Reprocessing)
    if st.session_state.query_result:
        st.subheader("Query Result:")
        try:
            if st.session_state.query_result['relevance'] == 'relevant':

                formatted_query = sqlparse.format(st.session_state.query_result["sql_query"], reindent=True, keyword_case='upper')
                # print(f"formatted_query\n{formatted_query}")
                with st.chat_message('ai'):
                    st.write("SQL Query")
                    st.code(formatted_query, language="sql")
                    st.session_state["chat_history"] += [AIMessage(formatted_query)]
                    st.write("Answer")
                    st.write(st.session_state.query_result["query_result"])
                    st.session_state["chat_history"] += [AIMessage(st.session_state.query_result["query_result"])]

        except Exception as e:
            st.error(f"Error displaying results: {e}")


if __name__ == "__main__":
    run()
