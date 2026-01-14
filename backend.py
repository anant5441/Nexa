import os
import asyncio
import threading
import tempfile
import requests
import aiosqlite
from typing import TypedDict, Annotated, Any, Dict, Optional

# Monkeypatch for langgraph compatibility with aiosqlite >= 0.22.0
if not hasattr(aiosqlite.Connection, "is_alive"):
    def is_alive(self):
        return self._thread.is_alive()
    aiosqlite.Connection.is_alive = is_alive

# Environment loading
from dotenv import load_dotenv

# LangChain / LangGraph / Google GenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ===== Checkpointer Import (Robust Handling) =====
try:
    # Try importing SQLite checkpointer (Requires: pip install langgraph-checkpoint-sqlite)
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    HAS_SQLITE = True
except ImportError:
    # Fallback to Memory checkpointer if SQLite lib is missing
    from langgraph.checkpoint.memory import MemorySaver
    HAS_SQLITE = False
    print("⚠️ WARNING: 'langgraph-checkpoint-sqlite' not found.")
    print("   Using in-memory storage. Chat history will be lost on restart.")
    print("   Run 'pip install langgraph-checkpoint-sqlite' to enable persistence.")

# GRPC setup for Gemini
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

load_dotenv()

# ===== Async Backend Loop Setup =====
_ASYNC_LOOP = asyncio.new_event_loop()

def _start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

_ASYNC_THREAD = threading.Thread(target=_start_background_loop, args=(_ASYNC_LOOP,), daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    return _submit_async(coro)

# ===== Model & Embeddings =====
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    batch_size=3,
)

# ===== Global State for Retrievers =====
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# ===== Tools =====

wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

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
        
        return {
            "first_num": first_num, 
            "second_num": second_num, 
            "operation": operation, 
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def list_github_prs(owner: str, repo: str, state: str = "open", per_page: int = 5) -> list[dict]:
    """
    List the latest pull requests for a GitHub repository.
    """
    token = os.getenv("GITHUB_TOKEN") 
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f'https://api.github.com/repos/{owner}/{repo}/pulls'
    params = {"state": state, "per_page": per_page}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        prs = []
        for pr in data:
            prs.append({
                "number": pr.get("number"),
                "title": pr.get("title"),
                "user": pr.get("user", {}).get("login"),
                "state": pr.get("state"),
                "url": pr.get("html_url"),
            })
        return prs
    except Exception as e:
        return [{"error": str(e)}]

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY not set in environment"}
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ===== MCP Client =====
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": r"C:\Users\Anant\Desktop\LangGraph\chatbot\venv1\Scripts\python.exe",
            "args": ["-u", r"C:\Users\Anant\Desktop\LangGraph\chatbot\mcp_server.py"],
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print(f"MCP Tools Error (ignoring): {e}")
        return []

mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools, calculator, rag_tool, list_github_prs]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# ===== Graph Definition =====

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful."
        )
    )
    messages = [system_message, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

async def _init_checkpointer():
    if HAS_SQLITE:
        conn = await aiosqlite.connect(database="chatbot.db")
        return AsyncSqliteSaver(conn)
    else:
        return MemorySaver()

# Initialize checkpointer on the background loop
checkpointer = run_async(_init_checkpointer())

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ===== Helper Functions for Frontend =====

async def _alist_threads():
    all_threads = set()
    
    # Handle difference between AsyncSqliteSaver (async) and MemorySaver (sync)
    if HAS_SQLITE:
        async for checkpoint in checkpointer.alist(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    else:
        # MemorySaver .list is synchronous
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
            
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})