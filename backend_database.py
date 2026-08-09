from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import sqlite3
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import random
from langgraph.prebuilt import ToolNode,tools_condition



load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)
## search tool
search_tool=DuckDuckGoSearchRun(region='us-en')


# calculator tool
@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """"
    
    perform a basic arithematic operations on two numbers.
    Supported operations:add,sub,mul,div
    """

    try:
        if operation=='add':
            result=first_num+second_num
        elif operation=='sub':
            result=first_num-second_num
        elif operation=='mul':
            result=first_num*second_num
        elif operation=='div':
            if second_num==0:
                return{'error':'division by zero is not allowed'}
            result=first_num/second_num
        else:
            return{'error':f"unsupported operation'{operation}'"}

        return{'first_num':first_num,'second_num':second_num,'operation':operation,'result':result}
    except Exception as e:
        return{"error":str(e)}


# stockprice tool

@tool
def get_stock_price(symbol:str)->dict:
    """"
    Fetch the latest stock price for given symbol (e.g. 'APPL'.'TSLA')
    using alpha vantage api key in the url
    """

    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=1CKDMBQ4CVMGM1YW"
    r=requests.get(url)
    return r.json()

## make tool list
tools=[get_stock_price,search_tool,calculator]

# make the llm tool-aware
llm_with_tools=llm.bind_tools(tools)

#state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

# graph node
def chat_node(state:ChatState):
    """"LLM node that may answer or request a tool call"""
    messages=state['messages']
    response=llm_with_tools.invoke(messages)
    return{'messages':[response]}

tool_node=ToolNode(tools)   # execute tool calls


#graph
#graph structure
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge("tools","chat_node")


#checkpointers
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