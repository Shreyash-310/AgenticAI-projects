from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from config import GOOGLE_API_KEY, LLM_MODEL, DB_PATH
from db_manager import DatabaseManager

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    api_key=GOOGLE_API_KEY
)

# Initialize database
db_manager = DatabaseManager(DB_PATH)


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


# Simple graph without checkpointing - we manage history ourselves
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile()

thread_id = '1'

if __name__ == '__main__':
    # Create thread if it doesn't exist
    if not db_manager.thread_exists(thread_id):
        db_manager.create_thread(thread_id, title="New Chat")
    
    # Load conversation history from database
    previous_messages = db_manager.get_messages(thread_id)
    title_set = False
    
    while True:
        user_message = input('Type Here: ')
        print(f"User : {user_message}")

        if user_message.strip().lower() in ['exit', 'quit', 'bye']:
            break
        
        # Add user message to in-memory history
        previous_messages.append(HumanMessage(content=user_message))
        
        # Set thread title to first message if not set
        if not title_set:
            title = user_message[:50] if len(user_message) <= 50 else user_message[:47] + "..."
            db_manager.update_thread_title(thread_id, title)
            title_set = True
        
        # Get AI response
        response = chatbot.invoke({'messages': previous_messages})
        ai_response = response['messages'][-1].content
        print(f"AI : {ai_response}")
        
        # Add AI response to in-memory history
        previous_messages.append(AIMessage(content=ai_response))
        
        # Save both to database
        db_manager.save_message(thread_id, 'user', user_message)
        db_manager.save_message(thread_id, 'assistant', ai_response)