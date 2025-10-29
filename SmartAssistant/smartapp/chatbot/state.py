from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any

class ChatbotState(BaseModel):
    """Represents the state of the chatbot flow."""

    query: Optional[str] = Field(
        default=None, 
        description="User's current query"
    )
    query_type: Optional[Literal["llm", "rag", "db"]] = Field(
        default=None, description="Type of query determined by classifier"
    )
    context: Optional[List[str]] = Field(
        default=None, description="Retrieved context for RAG"
    )
    answer: Optional[str] = Field(
        default=None, description="Final generated answer"
    )
    reasoning_trace: Optional[str] = Field(
        default=None, description="Explanation of reasoning"
    )
