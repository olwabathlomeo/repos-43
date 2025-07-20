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

# Load the model
with open("best_xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define input features
st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
loan_term = st.sidebar.number_input("Loan Term (in months)", min_value=6, max_value=480, value=360, step=6)
cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900, value=650)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, step=1000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, step=1000)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, step=1000)

# Encode inputs
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Create input DataFrame
input_data = {
    'no_of_dependents': [no_of_dependents],
    'education': [education_encoded],
    'self_employed': [self_employed_encoded],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'bank_asset_value': [bank_asset_value]
}

input_df = pd.DataFrame(input_data)

# Predict and display result
if st.button("Predict Loan Approval"):
    # Use .values to bypass feature name validation
    prediction = model.predict(input_df.values)[0]
    prediction_label = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.subheader(f"Loan Status: {prediction_label}")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df.values)

    st.subheader("🔍 SHAP Explanation (Why this prediction?)")
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0]), height=300)
