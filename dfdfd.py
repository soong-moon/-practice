import streamlit as st
from datetime import datetime

# 간단한 챗봇 응답 로직
def chatbot_response(user_input):
    if "안녕" in user_input:
        return "안녕하세요! 무엇을 도와드릴까요?"
    elif "날씨" in user_input:
        return "오늘 날씨는 정말 좋네요! ☀️"
    elif "시간" in user_input:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"현재 시간은 {now}입니다."
    else:
        return "죄송하지만, 무슨 말인지 잘 이해하지 못했어요. 다시 말씀해주시겠어요?"

# Streamlit UI
st.title("Streamlit 기반 챗봇")
st.subheader("챗봇과 대화해보세요!")

# 사용자 입력
user_input = st.text_input("메시지를 입력하세요:", "")

# 챗봇 응답 처리
if user_input:
    response = chatbot_response(user_input)
    st.write("🤖 챗봇:", response)

# 대화 로그를 보여주기 위한 상태 관리
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if user_input:
    st.session_state.conversation.append({"user": user_input, "bot": response})

st.subheader("대화 기록")
for message in st.session_state.conversation:
    st.write(f"👤 사용자: {message['user']}")
    st.write(f"🤖 챗봇: {message['bot']}")