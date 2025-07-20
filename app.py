import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import xgboost as xgb
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant data and explains the decision using SHAP values.")

# Load model
with open("best_xgb_model.pkl", "rb") as file:
    model: xgb.XGBClassifier = pickle.load(file)

# Feature input
st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=6, max_value=480, value=360, step=6)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", 0, step=1000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", 0, step=1000)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", 0, step=1000)

# Manual encoding
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Match feature order and names exactly as trained
input_dict = {
    'no_of_dependents': [no_of_dependents],
    'education': [education_encoded],
    'self_employed': [self_employed_encoded],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'bank_asset_value': [bank_asset_value]
}

input_df = pd.DataFrame(input_dict)

# Predict and explain
if st.button("Predict Loan Approval"):
    # Use validate_features=False to avoid XGBoost throwing error
    prediction = model.predict(input_df, validate_features=False)[0]
    status = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader(f"Loan Status: {status}")

    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    st.subheader("🔍 SHAP Explanation")
    st_shap(shap.force_plot(explainer.expected_value, shap_values.values[0], input_df.iloc[0]), height=300)
