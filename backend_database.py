# ============================================================
# IMPORTS
# ============================================================

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
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
# LOAD ENVIRONMENT VARIABLES
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

llm =ChatGroq(model="openai/gpt-oss-120b")


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# THREAD-BASED PDF RETRIEVERS
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available.
    """

    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    return None


# ============================================================
# PDF INGESTION
# ============================================================
def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF
    and store it for the specific chat thread.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Remove pages with no readable text
        docs = [
            doc for doc in docs
            if doc.page_content and doc.page_content.strip()
        ]

        if not docs:
            raise ValueError(
                "No readable text could be extracted from this PDF. "
                "The PDF may be scanned/image-based."
            )

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(docs)

        print(f"Documents loaded: {len(docs)}")
        print(f"Chunks created: {len(chunks)}")

        if not chunks:
            raise ValueError(
                "PDF produced no text chunks."
            )

        # Create FAISS vector store ONCE
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Store retriever for this thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        # Store metadata
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ============================================================
# SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic operations on two numbers.

    Supported operations:
    add, sub, mul, div
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
                    "error": "division by zero is not allowed"
                }

            result = first_num / second_num

        else:

            return {
                "error": f"unsupported operation '{operation}'"
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
# STOCK PRICE TOOL
# ============================================================

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol.

    Example:
    AAPL
    TSLA
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:

        return {
            "error": "ALPHAVANTAGE_API_KEY is not configured."
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
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
# RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve relevant information from the uploaded PDF
    for the current chat thread.

    Always provide the thread_id when calling this tool.
    """

    retriever = _get_retriever(thread_id)

    if retriever is None:

        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query
        }

    try:

        result = retriever.invoke(query)

        context = [
            doc.page_content
            for doc in result
        ]

        metadata = [
            doc.metadata
            for doc in result
        ]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": _THREAD_METADATA
                .get(str(thread_id), {})
                .get("filename")
        }

    except Exception as e:

        return {
            "error": str(e),
            "query": query
        }


# ============================================================
# TOOL LIST
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

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ChatState):
    """
    LLM node that may answer directly
    or request a tool call.
    """

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)


# START → CHAT

graph.add_edge(
    START,
    "chat_node"
)


# CHAT → TOOLS or END

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


# TOOLS → CHAT

graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    "chatbot_db.sqlite",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conn
)


# ============================================================
# COMPILE GRAPH
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# RETRIEVE ALL THREADS
# ============================================================

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = (
            checkpoint.config["configurable"]["thread_id"]
        )

        all_threads.add(thread_id)

    return list(all_threads)


# ============================================================
# CHECK WHETHER THREAD HAS DOCUMENT
# ============================================================

def thread_has_document(thread_id: str) -> bool:

    return str(thread_id) in _THREAD_RETRIEVERS


# ============================================================
# GET THREAD DOCUMENT METADATA
# ============================================================

def thread_document_metadata(thread_id: str) -> dict:

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )