import streamlit as st
from backend_database import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage,ToolMessage
import uuid
from backend_database import chatbot,ingest_pdf,retrieve_all_threads,thread_document_metadata
                              


# ============================================================
# Utility Functions
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ============================================================
# Session Setup
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# ============================================================
# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("AdBoT")

# New Chat
if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Search conversations
search = st.sidebar.text_input(
    "🔍 Search",
    placeholder="Search conversations..."
)

# Display conversations
for thread_id in st.session_state["chat_threads"]:

    # Get messages for this conversation
    messages = load_conversation(thread_id)

    # Default title
    title = "New Chat"

    # Get first user message as title
    for msg in messages:
        if isinstance(msg, HumanMessage):
            title = msg.content[:30]
            break

    # Apply search filter
    if search and search.lower() not in title.lower():
        continue

    # Show conversation title instead of thread ID
    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({
                "role": role,
                "content": msg.content
            })

        st.session_state["message_history"] = temp_messages

        st.rerun()

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# ============================================================
# Main UI
# ============================================================

st.title("AdBoT")


# Display conversation history

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input("Type here...")


if user_input:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    #CONFIG = {
       # "configurable": {
       #     "thread_id": st.session_state["thread_id"]
      #  }
   # }
    
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # --------------------------------------------------------
    # Stream AI response
    # --------------------------------------------------------

    with st.chat_message('assistant'):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):

                if isinstance(message_chunk,AIMessage):
                    yield message_chunk.content

        ai_message=st.write_stream(ai_only_stream())

#********************************Status writer**************************************************

     # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
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

    # --------------------------------------------------------
    # Save AI response to UI history
    # --------------------------------------------------------

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })