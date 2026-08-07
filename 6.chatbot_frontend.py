import streamlit as st

message_history=[]

for message in message_history:
     with st.chat_message(message['role']):
          st.text(message['content'])

# user message
with st.chat_message('user'):
    st.text('Hi')

# AI message
with st.chat_message('assistant'):
    st.text('How can i hepl you?')

with st.chat_message('user'):
    st.text('My name is Aditya')


# making chatbox

user_input=st.chat_input('Type your message here')

#minking user input to chatbox
if user_input:

    # first add message to message_history
    message_history.append({'role':'user','content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    message_history.append({'role':'assistant','content': user_input})
    with st.chat_message('assistant'):
            st.text(user_input)
# chat history storage
