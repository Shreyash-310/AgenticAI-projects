import os
from langchain_google_genai import ChatGoogleGenerativeAI
from chatbot.state import ChatbotState
# from state import ChatbotState
# LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=os.getenv('GOOGLE_API_KEY'))

from dotenv import load_dotenv
load_dotenv(dotenv_path='D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/.env')

def llm_node(state: ChatbotState) -> ChatbotState:
    """
    Handles open-domain or reasoning queries directly using Gemini.
    """
    try:

        prompt = f""" You are a helpful AI assistant. You will receive a conversation history and a new user query.

            Your job is to:
            1. Understand the context from the prior conversation history.
            2. Identify what the user is asking in the latest message.
            3. Answer naturally and helpfully, based on both:
            - The conversation history
            - The new user query

            Guidelines:
            - DO NOT repeat the conversation history in your response.
            - If the answer depends on prior messages, incorporate context smoothly.
            - If the user asks something unclear, ask a clarifying question.
            - If you do not know an answer, say so rather than guessing.
            - Keep responses concise unless asked for detail.


            New User Query:
            {state.query}

            Your Response: 
        """

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7,  api_key=os.getenv('GOOGLE_API_KEY'))
        response = llm.invoke(prompt)

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