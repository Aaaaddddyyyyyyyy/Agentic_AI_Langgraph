from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


conn = sqlite3.connect(
    database="chatbot_db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(
    checkpointer=checkpointer
)

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        all_threads.add(thread_id)

    return list(all_threads)