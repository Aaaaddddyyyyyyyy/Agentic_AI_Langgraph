from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.3-70b-versatile")


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

conn=sqlite3.connect(database='chatbot_db',check_same_thread=False)


checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)


for message_chunk,metadata in chatbot.stream(
    {'messages':[HumanMessage(content='what is the reciepe to make pasta?')]},
    config = {'configurable' : {'thread_id': 'thread1'}},
    stream_mode='messages'

):
    if message_chunk.content:
        print(message_chunk.content,end=" ",flush=True)

all_threads = set()
for checkpoint in checkpointer.list(None):
    all_threads.add(checkpoint.config['configurable']['thread_id'])

print(list(all_threads))