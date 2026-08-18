# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from backend_database import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf,
    set_current_thread,
    thread_document_metadata
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

import uuid


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_thread_id():

    return str(
        uuid.uuid4()
    )


def add_thread(
    thread_id
):

    if thread_id not in st.session_state[
        "chat_threads"
    ]:

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


def reset_chat():

    thread_id = generate_thread_id()


    st.session_state[
        "thread_id"
    ] = thread_id


    st.session_state[
        "message_history"
    ] = []


    add_thread(
        thread_id
    )


def load_conversation(
    thread_id
):

    state = chatbot.get_state(

        config={
            "configurable": {
                "thread_id":
                thread_id
            }
        }

    )


    return state.values.get(
        "messages",
        []
    )


# ============================================================
# SESSION STATE
# ============================================================

if "message_history" not in st.session_state:

    st.session_state[
        "message_history"
    ] = []


if "thread_id" not in st.session_state:

    st.session_state[
        "thread_id"
    ] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state[
        "chat_threads"
    ] = retrieve_all_threads()


if "ingested_docs" not in st.session_state:

    st.session_state[
        "ingested_docs"
    ] = {}


add_thread(
    st.session_state[
        "thread_id"
    ]
)


thread_id = str(
    st.session_state[
        "thread_id"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🤖 GoGo"
)


# ------------------------------------------------------------
# New Chat
# ------------------------------------------------------------

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


# ------------------------------------------------------------
# PDF Upload
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "📄 PDF"
)


uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_pdf:

    docs_for_thread = st.session_state[
        "ingested_docs"
    ].setdefault(
        thread_id,
        {}
    )


    if uploaded_pdf.name not in docs_for_thread:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True
        ) as status:

            try:

                summary = ingest_pdf(

                    uploaded_pdf.getvalue(),

                    thread_id=thread_id,

                    filename=uploaded_pdf.name
                )


                docs_for_thread[
                    uploaded_pdf.name
                ] = summary


                status.update(

                    label="✅ PDF indexed",

                    state="complete",

                    expanded=False
                )


                st.rerun()


            except Exception as e:

                status.update(

                    label="❌ PDF indexing failed",

                    state="error",

                    expanded=True
                )

                st.sidebar.error(
                    str(e)
                )


    else:

        st.sidebar.success(
            f"Using: {uploaded_pdf.name}"
        )


# ------------------------------------------------------------
# Current PDF information
# ------------------------------------------------------------

metadata = thread_document_metadata(
    thread_id
)


if metadata:

    st.sidebar.info(

        f"📄 **{metadata.get('filename')}**\n\n"

        f"Pages: {metadata.get('documents')}\n\n"

        f"Chunks: {metadata.get('chunks')}"
    )

else:

    st.sidebar.caption(
        "No PDF indexed for this chat."
    )


# ============================================================
# CONVERSATIONS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "💬 My Conversations"
)


search = st.sidebar.text_input(
    "🔍 Search",
    placeholder="Search conversations..."
)


for conversation_thread in reversed(
    st.session_state["chat_threads"]
):

    messages = load_conversation(
        conversation_thread
    )


    title = "New Chat"


    for msg in messages:

        if isinstance(
            msg,
            HumanMessage
        ):

            title = msg.content[:30]

            if len(msg.content) > 30:

                title += "..."

            break


    if search:

        if search.lower() not in title.lower():

            continue


    if st.sidebar.button(

        title,

        key=f"thread_{conversation_thread}",

        use_container_width=True
    ):

        st.session_state[
            "thread_id"
        ] = conversation_thread


        temp_messages = []


        for msg in messages:

            if isinstance(
                msg,
                HumanMessage
            ):

                role = "user"


            elif isinstance(
                msg,
                AIMessage
            ):

                role = "assistant"


            else:

                continue


            temp_messages.append({

                "role":
                role,

                "content":
                msg.content
            })


        st.session_state[
            "message_history"
        ] = temp_messages


        st.rerun()


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "🤖 GoGo"
)


st.caption(
    "LangGraph AI Assistant"
)


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Type your message..."
)


if user_input:

    # --------------------------------------------------------
    # Current thread
    # --------------------------------------------------------

    thread_id = str(
        st.session_state[
            "thread_id"
        ]
    )


    # --------------------------------------------------------
    # Set backend current thread
    # --------------------------------------------------------

    set_current_thread(
        thread_id
    )


    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append({

        "role":
        "user",

        "content":
        user_input
    })


    with st.chat_message(
        "user"
    ):

        st.write(
            user_input
        )


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    CONFIG = {

        "configurable": {

            "thread_id":
            thread_id
        },

        "metadata": {

            "thread_id":
            thread_id
        },

        "run_name":
        "chat_turn"
    }


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        status_holder = {
            "box": None
        }


        def ai_stream():

            for message_chunk, metadata in chatbot.stream(

                {
                    "messages": [

                        HumanMessage(
                            content=user_input
                        )

                    ]
                },

                config=CONFIG,

                stream_mode="messages"
            ):


                # --------------------------------------------
                # Tool message
                # --------------------------------------------

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )


                    if status_holder[
                        "box"
                    ] is None:

                        status_holder[
                            "box"
                        ] = st.status(

                            f"🔧 Using `{tool_name}`...",

                            expanded=True
                        )

                    else:

                        status_holder[
                            "box"
                        ].update(

                            label=
                            f"🔧 Using `{tool_name}`...",

                            state=
                            "running",

                            expanded=True
                        )


                # --------------------------------------------
                # AI tokens
                # --------------------------------------------

                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    content = (
                        message_chunk.content
                    )


                    if isinstance(
                        content,
                        str
                    ):

                        yield content


        # ----------------------------------------------------
        # Stream
        # ----------------------------------------------------

        ai_message = st.write_stream(
            ai_stream()
        )


        # ----------------------------------------------------
        # Finish tool status
        # ----------------------------------------------------

        if status_holder[
            "box"
        ] is not None:

            status_holder[
                "box"
            ].update(

                label=
                "✅ Tool finished",

                state=
                "complete",

                expanded=False
            )


    # --------------------------------------------------------
    # Save AI response
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append({

        "role":
        "assistant",

        "content":
        ai_message
    })