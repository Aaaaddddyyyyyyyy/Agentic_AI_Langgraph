from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio

load_dotenv()


# ============================================================
# State
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# LLM
# ============================================================

llm =ChatGroq(model="openai/gpt-oss-120b")


# ============================================================
# Search Tool
# ============================================================

@tool
def search_tool(query: str) -> str:
    """Search the web for information."""

    results = DDGS().text(
        query,
        max_results=5
    )

    return "\n".join(
        f"{result['title']}: {result['body']}"
        for result in results
    )


# ============================================================
# Calculator Tool
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
# Tools
# ============================================================

tools = [
    calculator,
    search_tool
]


# ============================================================
# LLM with Tools
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# Build Graph
# ============================================================

def build_graph():

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""

        messages = state["messages"]

        response = await llm_with_tools.ainvoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

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

    return graph.compile()


# ============================================================
# Main
# ============================================================

async def main():

    chatbot = build_graph()

    result = await chatbot.ainvoke({
        "messages": [
            HumanMessage(
                content=(
                    "Find the product of 1234 and 2 "
                    "and give the answer like a cricket commentator"
                )
            )
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())