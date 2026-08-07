from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

llm=ChatGroq(model="llama-3.3-70b-versatile")
def chat_node(state:ChatState):
    # take user query
    messages=state['messages']
    # send to llm
    response=llm.invoke(messages)
    #response store
    return{'messages':[response]}

checkpointer= MemorySaver()

graph= StateGraph(ChatState)

# add nodes

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

initial_state = {
    'messages':[HumanMessage(content='what is the capital of india?')]
}
chatbot.invoke(initial_state)['messages'][-1].content

thread_id='1'

## sking the chatting element of chatbot
while True:

    user_message = input('Type here :')
    print('User : ', user_message)

    if user_message.strip().lower() in  ['exit','bye','quit']:
        break
    
    config = {'configurable' : {'thread_id': thread_id}}
    response=chatbot.invoke({'messages':[HumanMessage(content=user_message)]},config=config)

    print('AI : ',response['messages'][-1].content)

