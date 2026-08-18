# ============================================================
# IMPORTS
# ============================================================

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

import sqlite3
import requests
import tempfile
import os


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# PDF RETRIEVERS
# ============================================================

_THREAD_RETRIEVERS = {}

_THREAD_METADATA = {}

CURRENT_THREAD_ID = None


# ============================================================
# CURRENT THREAD
# ============================================================

def set_current_thread(thread_id: str):
    global CURRENT_THREAD_ID

    CURRENT_THREAD_ID = str(thread_id)


# ============================================================
# GET RETRIEVER
# ============================================================

def _get_retriever(thread_id=None):

    if thread_id:
        return _THREAD_RETRIEVERS.get(
            str(thread_id)
        )

    if CURRENT_THREAD_ID:
        return _THREAD_RETRIEVERS.get(
            CURRENT_THREAD_ID
        )

    return None


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
):

    if not file_bytes:
        raise ValueError(
            "No PDF data received."
        )

    thread_id = str(thread_id)

    # --------------------------------------------------------
    # Create temporary PDF
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)

        temp_path = temp_file.name


    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            temp_path
        )

        documents = loader.load()


        # ----------------------------------------------------
        # Remove empty pages
        # ----------------------------------------------------

        documents = [
            doc
            for doc in documents
            if doc.page_content
            and doc.page_content.strip()
        ]


        if not documents:

            raise ValueError(
                "No readable text found in this PDF. "
                "The PDF may be scanned/image-based."
            )


        # ----------------------------------------------------
        # Split text
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )


        chunks = splitter.split_documents(
            documents
        )


        if not chunks:

            raise ValueError(
                "No text chunks were created."
            )


        # ----------------------------------------------------
        # Create FAISS
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )


        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            }
        )


        # ----------------------------------------------------
        # Store retriever
        # ----------------------------------------------------

        _THREAD_RETRIEVERS[
            thread_id
        ] = retriever


        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        _THREAD_METADATA[
            thread_id
        ] = {
            "filename": filename or "uploaded.pdf",
            "documents": len(documents),
            "chunks": len(chunks)
        }


        return {
            "filename": filename or "uploaded.pdf",
            "documents": len(documents),
            "chunks": len(chunks)
        }


    finally:

        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


# ============================================================
# WEB SEARCH
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# CALCULATOR
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:

    """
    Perform basic arithmetic.

    Supported operations:
    add
    sub
    mul
    div
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

                return {
                    "error":
                    "Division by zero is not allowed."
                }

            result = first_num / second_num

        else:

            return {
                "error":
                f"Unsupported operation: {operation}"
            }


        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }


    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# STOCK PRICE
# ============================================================

@tool
def get_stock_price(
    symbol: str
) -> dict:

    """
    Fetch latest stock price.
    """

    api_key = os.getenv(
        "ALPHAVANTAGE_API_KEY"
    )


    if not api_key:

        return {
            "error":
            "ALPHAVANTAGE_API_KEY is not configured."
        }


    url = (
        "https://www.alphavantage.co/query"
        "?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )


    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()


    except requests.RequestException as e:

        return {
            "error": str(e)
        }


# ============================================================
# PDF RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str
) -> dict:

    """
    Search the PDF uploaded in the current conversation.

    ALWAYS use this tool when the user asks a question
    about the uploaded PDF or its contents.
    """

    retriever = _get_retriever()


    if retriever is None:

        return {
            "error":
            "No PDF has been uploaded for this conversation."
        }


    try:

        docs = retriever.invoke(
            query
        )


        context = []


        for doc in docs:

            context.append({

                "text":
                doc.page_content,

                "page":
                doc.metadata.get(
                    "page",
                    "unknown"
                ),

                "source":
                doc.metadata.get(
                    "source",
                    ""
                )
            })


        return {

            "query": query,

            "context": context,

            "source_file":
            _THREAD_METADATA
            .get(
                CURRENT_THREAD_ID,
                {}
            )
            .get(
                "filename"
            )
        }


    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# TOOLS
# ============================================================

tools = [

    get_stock_price,

    search_tool,

    calculator,

    rag_tool

]


# ============================================================
# LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(
    state: ChatState
):

    messages = state["messages"]


    system_message = {
        "role": "system",
        "content": """
You are a helpful AI assistant.

You have access to these tools:

1. rag_tool
2. calculator
3. get_stock_price
4. web search

IMPORTANT PDF RULE:

If the user asks anything about an uploaded PDF,
document, report, book, notes, or its contents,
you MUST use rag_tool.

Do NOT answer PDF-related questions from your
general knowledge.

Use the retrieved PDF context to answer.

If no PDF is available, tell the user to upload
a PDF first.

For normal questions, answer normally or use
the appropriate tool.
"""
    }


    response = llm_with_tools.invoke(
        [
            system_message,
            *messages
        ]
    )


    return {
        "messages": [
            response
        ]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


graph.add_node(
    "chat_node",
    chat_node
)


graph.add_node(
    "tools",
    tool_node
)


graph.add_edge(
    START,
    "chat_node"
)


graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# SQLITE
# ============================================================

conn = sqlite3.connect(
    "chatbot_db.sqlite",
    check_same_thread=False
)


checkpointer = SqliteSaver(
    conn
)


# ============================================================
# COMPILE
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# RETRIEVE THREADS
# ============================================================

def retrieve_all_threads():

    all_threads = set()


    for checkpoint in checkpointer.list(None):

        thread_id = (
            checkpoint
            .config["configurable"]
            ["thread_id"]
        )

        all_threads.add(
            thread_id
        )


    return list(
        all_threads
    )


# ============================================================
# THREAD DOCUMENT METADATA
# ============================================================

def thread_document_metadata(
    thread_id: str
):

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )


# ============================================================
# THREAD DOCUMENT CHECK
# ============================================================

def thread_has_document(
    thread_id: str
):

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )