import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import pickle
from streamlit_shap import st_shap

# Load model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize SHAP
shap.initjs()

# App title and instructions
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts if a loan will be **Approved** or **Rejected** using XGBoost and explains the prediction with SHAP.")

# Sidebar input form
st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
loan_term = st.sidebar.slider("Loan Term (in months)", 12, 360, 120, step=12)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, value=500000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, value=0)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, value=100000)

# Prepare input DataFrame
input_data = {
    'no_of_dependents': [no_of_dependents],
    'education': [1 if education == "Graduate" else 0],
    'self_employed': [1 if self_employed == "Yes" else 0],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'bank_asset_value': [bank_asset_value]
}
input_df = pd.DataFrame(input_data)

# Ensure correct feature order
expected_features = [
    'no_of_dependents',
    'education',
    'self_employed',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'bank_asset_value'
]
input_df = input_df[expected_features]

# Predict and display result
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    prediction_label = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.subheader(f"Loan Status: {prediction_label}")

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    st.subheader("🔍 SHAP Explanation")
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df), height=300)

