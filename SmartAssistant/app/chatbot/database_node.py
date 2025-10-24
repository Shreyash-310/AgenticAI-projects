from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from chatbot.state import ChatbotState

def db_query_node(state: ChatbotState) -> ChatbotState:
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
