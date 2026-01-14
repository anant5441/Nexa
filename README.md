# Nexa Chatbot (LangGraph + Streamlit)

A persistent, multi-modal chatbot built with LangGraph and Streamlit. It uses Google's Gemini models for responses, local MCP servers for tool extensions, and stores chat state in a local SQLite database.

## Features
- **Persistent Memory**: Chat history is saved in a local SQLite database (`chatbot.db`).
- **Multi-Threaded**: Switch between different conversations easily using the sidebar.
- **MCP Integration**: Uses the Model Context Protocol (MCP) to extend capabilities (e.g., local arithmetic server).
- **RAG Capabilities**: Upload PDFs to chat with your documents using vector embeddings.
- **Smart Titles**: Automatically generates short titles for your conversations.
- **Tools**:
  - **Web Search**: DuckDuckGo search integration.
  - **Stock Price**: Alpha Vantage integration for financial data.
  - **Arithmetic**: Local MCP server for safe mathematical operations.
  - **RAG**: Document question-answering.

## Prerequisites
- Python 3.11+
- Google Gemini API key
- (Optional) Alpha Vantage API key (for stock prices)
- (Optional) GitHub Token (for PR listing)

## Setup

1. **Clone/Enter the project**:
   ```powershell
   cd LangGraph/chatbot
   ```

2. **Set up Virtual Environment**:
   ```powershell
   python -m venv venv1
   .\venv1\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the `chatbot/` directory:
   ```env
   GOOGLE_API_KEY=your_google_gemini_key
   ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
   GITHUB_TOKEN=your_github_token
   ```

## Running the App

Start the Streamlit frontend:
```powershell
python -m streamlit run frontend.py
```

The application will open in your browser at `http://localhost:8501`.

## Project Structure

- `frontend.py`: The Streamlit User Interface. Handles chat rendering and streaming.
- `backend.py`: The LangGraph agent definition.
  - Includes a compatibility patch for `aiosqlite`.
  - Configures the MCP client.
- `mcp_server.py`: A local MCP server providing arithmetic tools (Add/Sub/Mul/Div).
- `chatbot.db`: Local SQLite database for chat history.

## Troubleshooting

### Startup Errors
If you see an error about `is_alive` not being found on `aiosqlite.Connection`:
- This is a known incompatibility between `langgraph-checkpoint-sqlite` and `aiosqlite` v0.22+.
- **Fix**: The included `backend.py` contains a monkeypatch to resolve this automatically. Ensure you are running the latest version of the code.

### MCP Errors
If you see connection errors related to MCP:
- Ensure `mcp_server.py` exists in the same directory.
- The backend is configured to run this server using the current Python environment.
