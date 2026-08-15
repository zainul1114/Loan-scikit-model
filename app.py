import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Portal", page_icon="🏦")

@st.cache_resource
def load_model():
    return joblib.load("loan_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("🏦 Loan Underwriting Portal")

with st.form("underwriting_form"):
    income = st.number_input("Annual Income ($)", min_value=10000, value=65000)
    credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=680)
    employment = st.selectbox("Employment Status", options=["employed", "unemployed", "self-employed"])
    submit_btn = st.form_submit_button("Run Risk Check")

if submit_btn:
    input_payload = pd.DataFrame([{"income": income, "credit_score": credit_score, "employment": employment}])
    prediction = model.predict(input_payload)
    prob = model.predict_proba(input_payload)[0][1]

    if prediction[0] == 1:
        st.success(f"🎉 Approved! Approval Probability: {prob * 100:.1f}%")
    else:
        st.error(f"❌ Declined. Approval Probability: {prob * 100:.1f}%")

