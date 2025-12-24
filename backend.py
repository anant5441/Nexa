from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',temperature=0
)


search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def list_github_prs(owner:str, repo:str, state: str ="open", per_page:int =5, ):
    """
    List the latest pull requests for a GitHub repository.
    Args:
        owner:GitHub org or username (e.g., "langgraph-ai")
        repo:Repository name (e.g., "langgraph")
        state: "open", "closed", or "all"
        per_page: Number of PRs to fetch (max 100)
    Returns:
        simplified list of PR info dictionaries.
    """
    token = os.getenv("GITHUB_TOKEN") # optional, for higher rate limits
    headers = {
        "Accept":"application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url =f'https://api.github.com/repos/{owner}/{repo}/pulls'
    params = {
        "state":state,
        "per_page":per_page, 
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    # Assumes GitHub always returns valid JSON with a list of PR objects
    response.raise_for_status()
    data = response.json()
    # Assumes these keys ("number", "title", "user'", "state", "html_url") exist
    prs=[]
    for pr in data:
        prs.append(
            {
                "number":pr["number"],
                "title":pr["title"],
                "user":pr["user"]["login"],
                "state":pr["state"],
                "url":pr["html_url"],
            }
        )
        return prs

SERVERS = {
    "github" : {
        "transport":"studio",
        "command":"/usr/bin/python3",
        "args":[
            "/path/to/github_mcp_server.py"
        ]
    }
}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not set in environment")
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)