import streamlit as st
from backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid
import queue



# ===== Utility functions =====
def generate_thread_id():
    return str(uuid.uuid4())

def save_chat_title(thread_id):
    if thread_id in st.session_state['generated_titles']:
        return  # Title already exists

    # Get the first user message
    first_user_message = next(
        (msg['content'] for msg in st.session_state['message_history']
            if msg['role'] == 'user'),
        None
    )

    if not first_user_message:
        return

    from langchain_google_genai import ChatGoogleGenerativeAI
    title_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    title_prompt = f"Create a very short (max 5 words) chat title based on this message which is very effictive related (example: I say hello then title become Greeting message).Give only of 3 words: '{first_user_message}'"
    title = title_model.invoke(title_prompt).content.strip()

    st.session_state['generated_titles'][thread_id] = title


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
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        return messages if messages else []
    except Exception as e:
        st.error(f"Could not load conversation for thread {thread_id}: {e}")
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

add_thread(st.session_state['thread_id'])

# ===== Sidebar =====
st.sidebar.title('Nexa')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    display_name = st.session_state['generated_titles'].get(thread_id, thread_id)
    if st.sidebar.button(display_name):
        save_chat_title(st.session_state['thread_id'])  # Save current chat title before switching
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        st.session_state['message_history'] = [
            {'role': 'user' if isinstance(msg, HumanMessage) else 'assistant', 'content': msg.content}
            for msg in messages
        ]

# ===== Main Chat UI =====
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    thread_id = st.session_state['thread_id']

    # Add user's message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # CONFIG = {'configurable': {'thread_id': thread_id}}
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
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
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

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Lazily create & update the SAME status container when any tool runs
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

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )