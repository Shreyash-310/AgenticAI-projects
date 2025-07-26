# importing a necessary library
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='../.env')

from langgraph.graph import StateGraph,MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

class chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            temperature = 0,
            groq_api_key = os.getenv("groq_api_key"),
            model_name = os.getenv("llama_model_name2")
            )
        self.tavily_api_key = os.getenv('tavily_api_key')
        
    def call_tool(self):
        tool = TavilySearchResults(
            # self.tavily_api_key,
            max_results=2)
        tools = [tool]
        self.tool_node = ToolNode(tools=[tool])
        self.llm_with_tool=self.llm.bind_tools(tools)
        
    def call_model(self,state: MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}
    
    def router_function(self,state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
                
    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent",self.router_function,{"tools": "tools", END: END})
        workflow.add_edge("tools", 'agent')
        self.app = workflow.compile()
        return self.app
        
if __name__=="__main__":
    mybot=chatbot()
    workflow=mybot()
    response=workflow.invoke({"messages": ["who is a current chief minister of Maharashtra, India?"]})
    print(response['messages'][-1].content)