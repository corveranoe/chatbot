import streamlit as st
from chatbot import predict_class, get_response, intents

st.title("🤖 Bienvenido a tu Asistente Virtual")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
        st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"})
        st.session_state.first_message = False

if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    intents_list = predict_class(prompt)
    response = get_response(intents_list, intents)
    
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
