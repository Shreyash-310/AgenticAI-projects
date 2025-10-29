from langgraph.graph import StateGraph, END
from chatbot.state import ChatbotState
from chatbot.classifier_node import classify_query_node
from chatbot.rag_node import rag_node
from chatbot.database_node import db_query_node
from chatbot.llm_node import llm_node

def router(state: ChatbotState) -> str:
    """
    Router node — decides which branch to take based on query_type.
    Returns one of: 'rag', 'db', 'llm'
    """
    if state.query_type == "rag":
        return "rag"
    elif state.query_type == "db":
        return "db"
    else:
        return "llm"


def build_chatbot_graph() -> StateGraph:
    """
    Build the chatbot LangGraph pipeline with classification + routing.
    Compatible with latest LangGraph API.
    """
    graph = StateGraph(ChatbotState)

    # --- Define nodes ---
    graph.add_node("classify", classify_query_node)
    graph.add_node("rag", rag_node)
    graph.add_node("db", db_query_node)
    graph.add_node("llm", llm_node)

    # --- Edges ---
    # First step always goes to classification
    graph.set_entry_point("classify")

    # Conditional routing
    graph.add_conditional_edges(
        "classify",
        router,  # function that returns key: rag / db / llm
        {
            "rag": "rag",
            "db": "db",
            "llm": "llm",
        },
    )

    # --- End edges ---
    graph.add_edge("rag", END)
    graph.add_edge("db", END)
    graph.add_edge("llm", END)

    graph.set_entry_point("classify")

    app = graph.compile()

    return app

# Initialize chatbot graph
chatbot_graph = build_chatbot_graph()

# Example invocation helper
def run_chatbot(query: str) -> ChatbotState:
    """
    Run the query through the LangGraph pipeline.
    Returns the final ChatbotState with answer, context, reasoning_trace.
    """
    initial_state = ChatbotState(query=query)
    result_state = chatbot_graph.invoke(initial_state)
    return result_state

if __name__ == '__main__':

    def run_chatbot(query: str) -> ChatbotState:
        """
        Run the query through the LangGraph pipeline.
        Returns the final ChatbotState with answer, context, reasoning_trace.
        """
        initial_state = ChatbotState(query=query)
        result_state = chatbot_graph.invoke(initial_state)
        return result_state
    
    response = run_chatbot(query = 'what is survivor annunuity percentage for the plan from document?')
    print(f"response => {response}")