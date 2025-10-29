import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .state import ChatbotState
from typing import Literal

from dotenv import load_dotenv
load_dotenv(dotenv_path='D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/.env')

# Structured output schema for LLM classification
class QueryClassification(BaseModel):
    query_type: Literal["llm", "rag", "db"] = Field(
        ..., description="Type of query (llm, rag, or db)")
    reasoning: str = Field(..., description="Explanation of why this classification was made")

# Initialize model (replace with your working one, e.g., gemini-2.5-pro)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             temperature=0,
                             api_key=os.getenv('GOOGLE_API_KEY'))

# Define parser
parser = PydanticOutputParser(pydantic_object=QueryClassification)

# Define prompt
prompt = PromptTemplate(
    template=(
        "You are a query classification assistant. "
        "Given a user question, determine if it should be handled by:\n"
        "- 'llm': for general world knowledge or reasoning questions.\n"
        "- 'rag': for document-based or semantic search questions.\n"
        "- 'db': for database or structured data retrieval queries.\n\n"
        "Return a structured JSON matching this schema:\n{format_instructions}\n\n"
        "Question: {query}"
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def classify_query_node(state: ChatbotState) -> ChatbotState:
    """Classify query into llm, rag, or db using Gemini and structured output."""
    try:
        input_prompt = prompt.format_prompt(query=state.query)
        response = llm.invoke(input_prompt.to_string())
        parsed = parser.parse(response.content)

        state.query_type = parsed.query_type
        state.reasoning_trace = parsed.reasoning
        return state

    except Exception as e:
        state.query_type = "llm"  # fallback
        state.reasoning_trace = f"Classification failed: {e}"
        return state

def test_query_classification():
    queries = [
        "Summarize the uploaded research paper on AI.",
        "List all customers who purchased more than $500 last month.",
        "Who is the Prime Minister of Canada?"
    ]

    for q in queries:
        state = ChatbotState(query=q)
        print(f"\n---\nQuery: {q}")
        updated_state = classify_query_node(state)
        print(f"Predicted Type: {updated_state.query_type}")
        print(f"Reasoning: {updated_state.reasoning_trace}")

if __name__ == '__main__':
    test_query_classification()