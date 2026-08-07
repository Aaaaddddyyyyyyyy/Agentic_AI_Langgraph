import streamlit as st

# user message
with st.chat_message('user'):
    st.text('Hi')

# AI message
with st.chat_message('assistant'):
    st.text('How can i hepl you?')