import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from chatbot.state import ChatbotState
from state import ChatbotState
# LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=os.getenv('GOOGLE_API_KEY'))

from dotenv import load_dotenv
load_dotenv(dotenv_path='D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/.env')

def llm_node(state: ChatbotState) -> ChatbotState:
    """
    Handles open-domain or reasoning queries directly using Gemini.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7,  api_key=os.getenv('GOOGLE_API_KEY'))
        response = llm.invoke(state.query)

        state.answer = response.content
        state.reasoning_trace = "Used Gemini directly for general reasoning or world knowledge question."
        return state

    except Exception as e:
        state.answer = f"LLM node failed: {e}"
        state.reasoning_trace = "LLM node could not generate an answer."
        return state

if __name__ == '__main__':
    
    state = ChatbotState(query="Who is the current CEO of OpenAI?", query_type="llm")
    updated = llm_node(state)
    print(f"Answer:\n{updated.answer}\nReasoning: {updated.reasoning_trace}")