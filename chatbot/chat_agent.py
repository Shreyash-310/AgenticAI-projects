from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv
load_dotenv('../.env')

from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    api_key = os.getenv('GOOGLE_API_KEY')
)

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages':[response]}

checkpointer = InMemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = '1'

if __name__ == '__main__':

    while True:
        user_message = input('Type Here: ')
        print(f"User : {user_message}")

        if user_message.strip().lower() in ['exit','quit','bye']:
            break
        config = {'configurable':{'thread_id':thread_id}}
        response = chatbot.invoke({'messages':[HumanMessage(content=user_message)]}, config=config)
        print(f"AI : {response['messages'][-1].content}")