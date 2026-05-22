import streamlit as st
import pandas as pd
import joblib
import os

# 페이지 설정
st.set_page_config(page_title="토마토 착과율 예측", layout="centered")

st.title("🍅 토마토 착과율 예측")
st.markdown("---")

# 모델 로드
@st.cache_resource
def load_model():
    model_path = "tomato_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()
    model = joblib.load(model_path)
    return model

try:
    rf_model = load_model()
except Exception as e:
    st.error(f"모델 로드 실패: {str(e)}")
    st.stop()

# 입력 받기
st.subheader("환경 정보 입력")

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input(
        "내부온도 (°C)",
        value=20.0,
        min_value=-10.0,
        max_value=50.0,
        step=0.1
    )

with col2:
    humidity = st.number_input(
        "내부습도 (%)",
        value=60.0,
        min_value=0.0,
        max_value=100.0,
        step=0.1
    )

with col3:
    soil_temp = st.number_input(
        "지온 (°C)",
        value=20.0,
        min_value=-10.0,
        max_value=50.0,
        step=0.1
    )

st.markdown("---")

# 예측 버튼
if st.button("착과율 예측", type="primary"):
    # DataFrame으로 변환
    input_data = pd.DataFrame(
        [[temp, humidity, soil_temp]], 
        columns=['내부온도', '내부습도', '지온']
    )
    
    # 예측
    predicted = rf_model.predict(input_data)
    
    # 결과 표시
    st.success(f"## 예측 착과율: {predicted[0]:.1f}%")
    
    # 입력 값 요약
    with st.expander("입력 값 상세"):
        st.write(f"- 내부온도: {temp}°C")
        st.write(f"- 내부습도: {humidity}%")
        st.write(f"- 지온: {soil_temp}°C")
