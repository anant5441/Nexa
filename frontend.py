import streamlit as st
import uuid
import queue
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import backend logic
from backend import (
    chatbot, 
    retrieve_all_threads, 
    submit_async_task, 
    ingest_pdf, 
    thread_document_metadata
)

# ===== Utility functions =====
def generate_thread_id():
    return str(uuid.uuid4())

def save_chat_title(thread_id):
    if thread_id in st.session_state["generated_titles"]:
        return

    first_user_message = next(
        (
            msg["content"]
            for msg in st.session_state["message_history"]
            if msg["role"] == "user"
        ),
        None,
    )

    if not first_user_message:
        return

    try:
        # Using a separate lightweight instance for titles to avoid interference
        title_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        title_prompt = (
            "Create a very short (max 3 words) chat title that is highly relevant. "
            f"Message: '{first_user_message}'"
        )
        title = title_model.invoke(title_prompt).content.strip()
    except Exception:
        title = first_user_message[:20]

    st.session_state["generated_titles"][thread_id] = title


def reset_chat():
    save_chat_title(st.session_state['thread_id'])  # Save old chat title before resetting
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    try:
        # Fetch state synchronously using the backend helper (implicit via chatbot.get_state)
        # Note: chatbot.get_state is actually synchronous in compiled LangGraph 
        # unless configured otherwise, but if it needs async db access, we might need a wrapper.
        # However, checking LangGraph docs, get_state is sync-compatible if checkpointer allows.
        # Since we use AsyncSqliteSaver, we should use a method that bridges it, 
        # but LangGraph's CompiledGraph exposes sync wrappers. 
        # If this fails, we wrap it in a run_async from backend.
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        return messages if messages else []
    except Exception as e:
        # Fallback if get_state fails or is empty
        return []
    
# ===== Initialize session state =====
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
if 'generated_titles' not in st.session_state:
    st.session_state['generated_titles'] = {}
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state['thread_id'])
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ===== Sidebar =====
st.sidebar.title("Nexa")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

# Check for existing docs in backend metadata even if session state was cleared
backend_meta = thread_document_metadata(thread_key)
if backend_meta:
    # Sync backend meta to frontend state
    thread_docs[backend_meta.get("filename")] = backend_meta

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks, {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"]
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info("PDF already indexed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True):
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                st.sidebar.success("PDF indexed successfully")
            except Exception as e:
                st.sidebar.error(f"Failed to index PDF: {e}")

st.sidebar.subheader("Past Conversations")

# Reverse list to show newest first
for tid in st.session_state["chat_threads"][::-1]:
    label = st.session_state["generated_titles"].get(tid, tid)
    if st.sidebar.button(label, key=f"thread-{tid}"):
        save_chat_title(st.session_state["thread_id"])
        st.session_state["thread_id"] = tid
        messages = load_conversation(tid)
        st.session_state["message_history"] = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ]
        st.rerun()

# ===== Main Chat UI =====

st.title("Multi Utility Chatbot")
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Ask about your document or use tools...')

if user_input:
    # Add user's message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant's response
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            # Queue to bridge backend async thread and frontend sync generator
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    # 'chatbot.astream' returns an async iterator
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            # Submit the async task to the backend loop
            submit_async_task(run_stream())

            # Yield from queue in the main thread
            while True:
                item = event_queue.get()
                if item is None:
                    break
                
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Update UI for Tools
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens (AIMessage)
                if isinstance(message_chunk, AIMessage):
                    # Only yield if there is content (ignores tool request chunks which are empty)
                    if message_chunk.content:
                        content = message_chunk.content
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    yield item.get("text", "")
                                elif isinstance(item, str):
                                    yield item
                        else:
                            yield content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize status box if a tool was used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )