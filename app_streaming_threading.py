import streamlit as st
from chatbot_mem import chatbot
from langchain_core.messages import HumanMessage
import uuid


# ============================================================
# Utility Functions
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id, "New Chat")

    st.session_state["message_history"] = []


def add_thread(thread_id, title="New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title


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
    st.session_state["chat_threads"] = {}


add_thread(st.session_state["thread_id"])


# ============================================================
# Sidebar UI
# ============================================================

st.sidebar.title("GoGo")


if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("My Conversations")


for thread_id, title in st.session_state["chat_threads"].items():

    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

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


# ============================================================
# Main UI
# ============================================================

st.title("GoGo")


# Display conversation history

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input("Type here")


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

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    current_thread = st.session_state["thread_id"]


    # --------------------------------------------------------
    # Generate chat title
    # --------------------------------------------------------

    if st.session_state["chat_threads"][current_thread] == "New Chat":

        title = user_input[:30]

        if len(user_input) > 30:
            title += "..."

        st.session_state["chat_threads"][current_thread] = title


    # --------------------------------------------------------
    # AI response - Streaming
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            )
        )


    # --------------------------------------------------------
    # Save AI response
    # --------------------------------------------------------

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })