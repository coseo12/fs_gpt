import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📜",
)

# 메시지 저장소
if "messages" not in st.session_state:
    st.session_state["messages"] = []    

# 채팅 메시지 출력
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

# 캐싱한 채팅 메시지 출력
for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

# 채팅 입력
message = st.chat_input("Send a message to AI")

# 입력된 채팅 메시지 출력
if message:
    send_message(message, 'human')
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)