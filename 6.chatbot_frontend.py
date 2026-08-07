import streamlit as st


# session state -> dict-> not erasing the history of particular session
if 'message_history' not in st.session_state:
     st.session_state['message_history']=[]


# loading conversation history
for message in st.session_state['message_history']:
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
    st.session_state['message_history'].append({'role':'user','content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    st.session_state['message_history'].append({'role':'assistant','content': user_input})
    with st.chat_message('assistant'):
            st.text(user_input)
# chat history storage
