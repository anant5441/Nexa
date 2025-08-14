# Nexa Chatbot (LangGraph + Streamlit)

A simple, persistent chatbot built with LangGraph and Streamlit. It uses Google's Gemini models for responses and stores chat state in a local SQLite database so you can revisit previous conversations.

## Features
- Multi-threaded conversations (thread list in sidebar)
- **ResumeChat Ready** - Switch between previous conversations seamlessly
- **SQLite Database Storage** - Persistent chat history with automatic state management
- Persistent state via SQLite (`chatbot.db`)
- Automatic short titles for conversations
- `.env`-based API key management (ignored by Git)

## Prerequisites
- Python 3.10+
- Google Gemini API key
- (Optional) Groq API key

## Setup
1. Navigate to the project folder:
   ```bash
   cd LangGraph/chatbot
   ```

2. Create and activate a virtual environment:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     ```
   - macOS/Linux (bash):
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -U streamlit langgraph langchain-google-genai google-generativeai langchain-groq python-dotenv
   ```

4. Create a `.env` file in `chatbot/` with your keys:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key
   # Optional, only if you plan to use Groq models anywhere
   GROQ_API_KEY=your_groq_api_key
   ```

   Note: `.env`, `__pycache__/`, and `*.pyc` are ignored by Git via `chatbot/.gitignore`.

## Run
From the `chatbot` directory, start the UI:
```bash
streamlit run frontend.py
```

- The app sidebar shows existing chats and lets you create a new one.
- Conversations are saved in `chatbot.db`. To fully reset, stop the app and remove `chatbot.db`.

## Files
- `backend.py`: LangGraph pipeline and SQLite checkpointer. Uses `gemini-2.0-flash`.
- `frontend.py`: Streamlit UI, chat threads, and title generation (uses `gemini-2.5-flash` for short titles).
- `chatbot.db`: Local SQLite database (auto-created at runtime).
- `.gitignore`: Ensures `.env` and Python cache files are not tracked.

## Troubleshooting
- If the app complains about missing keys, ensure `.env` exists and contains `GOOGLE_API_KEY` (and optionally `GROQ_API_KEY`), then restart the app.
- If you change environment variables, restart the terminal or re-activate the virtual environment so they reload.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using modern web technologies**
