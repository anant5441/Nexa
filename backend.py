from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

import os   
api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY not set in environment"

genai.configure(api_key=api_key)

import os
model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
)
model1 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

checkpointer=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)