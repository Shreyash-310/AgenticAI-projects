import os
import streamlit as st
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

class sql_agent:
    def __init__(self, database_path:str):
        """
        Initiate the SQL agent, llm, queey_agent
        Args:
            database_path : provide the absolute path of the database
        """
        self.db = SQLDatabase.from_uri(f"sqlite:///{database_path}")

        self.llm = ChatGroq(
            temperature = 0,
            groq_api_key = os.getenv("groq_api_key"),
            model_name = os.getenv("llama_model_name")
            )
        
        self.query_agent = create_sql_agent(self.llm, db=self.db,
                                            agent_type='openai-tools',
                                            verbose=False,
                                            agent_executor_kwargs={"return_intermediate_steps":True})

    def get_query(self,nlp_query:str):
        """
        Returns the output for the NLP Query
        Args:
            nlp_query : input NLP query related to database asked by the user
        Returns:
            sql_query : 
            ans :
        """
        response = self.query_agent.invoke(
            {"input":nlp_query}
            )
        
        queries = []
        for (log,output) in response['intermediate_steps']:
            if log.tool == 'sql_db_query':
                queries.append(log.tool_input)
        
        return queries[-1]['query'], response['output']

if __name__ == "__main__":

    header = st.empty()
    header.header("Chat with Database")

    db_path = st.sidebar.text_input("Enter the absolute path of the database:")
    if db_path:
        try:
            sql_query_agent = sql_agent(database_path=db_path)
            if sql_query_agent.db.dialect == "sqlite":
                st.sidebar.success(f"Connected to the database {os.path.basename(db_path)}")

                header.header(f"Chat with {os.path.basename(db_path)}")

            question = st.text_input("Enter your question related to the database")

            if question:
                with st.spinner("Processing....."):
                    query, answer = sql_query_agent.get_query(question)
                    st.success("Done")
                st.write("SQL Query")
                st.write(query)
                st.write("Answer")
                st.write(answer)

        except sqlite3.Error as e:
            st.sidebar.error(f"Error connecting to the database : {e}")
    else:
        st.sidebar.info("Please enter the path to your SQLite Database.")

# D:\GenAI-Practice\AgenticAI-Projects\Text2SQL\data\sql\