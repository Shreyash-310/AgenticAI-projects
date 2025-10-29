from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from chatbot.state import ChatbotState
from chatbot.nlp2sql import create_workflow
import os

def db_query_node(state: ChatbotState) -> ChatbotState:
    """
    Handles structured data queries via LangChain SQL Agent.
    Executes natural language questions on database and updates the state.
    """
    try:
        db_pth = 'D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/smartapp/data/db_uploads/'
        files = [file for file in os.listdir(db_pth) if file.endswith('.db')]
        if files:
            sql_app = create_workflow(os.path.join(db_pth,files[0]))
            query_result = sql_app.invoke(
                {"question": state.query, "attempts": 0},
                config={"configurable": {"current_user_id": "1"}}
                )
            
            # Update chatbot state
            if query_result['relevance'] == 'relevant':
                print(f"SQL Query:\n {query_result["sql_query"]}, \nAnswer:\n{query_result["query_result"]}")
                # state.answer = query_result["query_result"]
            elif query_result['relevance'] == 'not_relevant':
                print(f"Answer:\n{query_result["query_result"]}")

            state.answer = query_result["query_result"]
            state.reasoning_trace = (
                f"Used SQL Agent to execute query derived from user question."
            )
            
        return state

    except Exception as e:
        state.answer = f"Error during DB query: {e}"
        state.reasoning_trace = "DB Node failed to process query."
        
        return state    

def db_query_node1(state: ChatbotState) -> ChatbotState:
    """
    Handles structured data queries via LangChain SQL Agent.
    Executes natural language questions on database and updates the state.
    """
    try:
        # Load database
        db = SQLDatabase.from_uri("sqlite:///D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/app/data/uploads/bank_domain.db")

        # Create toolkit for SQL agent
        toolkit = SQLDatabaseToolkit(db=db, llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro"))

        # Create the SQL agent
        agent_executor = create_sql_agent(
            llm=toolkit.llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",  # works well with Gemini too
        )

        # Run query
        query = state.query
        response = agent_executor.invoke({"input": query})

        # Update chatbot state
        state.answer = response["output"]
        state.reasoning_trace = (
            f"Used SQL Agent on 'sample.db' to execute query derived from user question."
        )
        return state

    except Exception as e:
        state.answer = f"Error during DB query: {e}"
        state.reasoning_trace = "DB Node failed to process query."
        return state
