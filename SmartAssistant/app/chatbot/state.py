from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Any

class ChatbotState(BaseModel):
    """Represents the state of the chatbot flow."""
    
    query: str = Field(..., description="User's current query")
    query_type: Optional[Literal["llm", "rag", "db"]] = Field(
        default=None, description="Type of query determined by classifier")
    context: Optional[List[str]] = Field(
        default=None, description="Retrieved context or documents for RAG")
    answer: Optional[str] = Field(
        default=None, description="Final generated answer for user")
    reasoning_trace: Optional[str] = Field(
        default=None, description="Explanation of reasoning or tool selection")
    memory: Optional[Any] = Field(
        default=None, description="Conversation memory (optional future use)")
