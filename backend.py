from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

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

# Initialize the SQLite database for state persistence
conn=sqlite3.connect(database='chatbot.db',check_same_thread=False) #check_same_thread=False means that this database connection can be used across multiple threads
#sqlite works only with single thread
checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)