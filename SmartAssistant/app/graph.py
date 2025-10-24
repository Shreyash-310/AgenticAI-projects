from langgraph.graph import StateGraph, END
from chatbot.state import ChatbotState
from chatbot.classifier_node import classify_query_node
from chatbot.rag_node import rag_node
from chatbot.database_node import db_query_node
from chatbot.llm_node import llm_node

def build_chatbot_graph() -> StateGraph:
    """
    Constructs the chatbot LangGraph pipeline:
    1. Classify query → decides route: rag / db / llm
    2. Route to appropriate node
    3. Update shared Pydantic state
    """

    # Initialize state graph
    graph = StateGraph()

    # --- Nodes ---
    graph.add_node("classify", classify_query_node)
    graph.add_node("rag", rag_node)
    graph.add_node("db", db_query_node)
    graph.add_node("llm", llm_node)

    # --- Routing edges from classify node ---
    graph.add_edge("classify", "rag", condition=lambda s: s.query_type == "rag")
    graph.add_edge("classify", "db", condition=lambda s: s.query_type == "db")
    graph.add_edge("classify", "llm", condition=lambda s: s.query_type == "llm")

    # --- End edges ---
    graph.add_edge("rag", END)
    graph.add_edge("db", END)
    graph.add_edge("llm", END)

    return graph

# Initialize the graph
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
