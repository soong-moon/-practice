import streamlit as st

st.title("스파르타코딩클럽 AI 8기 예제")
st.header("지금 스트림릿 예제를 만들고 있습니다")
st.text("지금 잘 따라가고 있습니다")

st.markdown("### h3 글씨를 표현한다")
st.latex("E=mc^2")

if st.button("버튼을 클릭하세요"):
    st.write("버튼이 눌렸다")

agree_box = st.checkbox("동의할래요?")
if agree_box is True :
    st.write("동의했어요")

volume = st.slider("음악 볼륨",0,100,50)
st.write("음악 볼륨은"+str(volume)+"입니다")

gender = st.radio ("성별을 선택하시오",["남자","여자"])
st.write("성별은 "+gender+"입니다.")