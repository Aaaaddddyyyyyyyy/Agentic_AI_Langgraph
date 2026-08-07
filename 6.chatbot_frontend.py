import streamlit as st

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
    with st.chat_message('user'):
        st.text(user_input)