import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

# Set page configuration
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.write("This app predicts loan approval using a trained XGBoost model and shows SHAP explanations.")

# Load the trained XGBoost model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define user input form
st.sidebar.header("Applicant Input Features")

def get_user_input():
    income_annum = st.sidebar.number_input("Annual Income (in Ksh)", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
    cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 700)
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    asset_value = st.sidebar.number_input("Total Asset Value", min_value=0)

    education_binary = 1 if education == "Graduate" else 0
    self_employed_binary = 1 if self_employed == "Yes" else 0

    data = {
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "cibil_score": cibil_score,
        "education": education_binary,
        "self_employed": self_employed_binary,
        "asset_value": asset_value
    }

    return pd.DataFrame([data])

input_df = get_user_input()

# Make prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    prediction_label = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader("Prediction Result:")
    st.success(f"The loan is likely to be: **{prediction_label}**")

    # SHAP explanation
    st.subheader("🔍 Feature Impact (SHAP Explanation)")

    # Create explainer and shap values
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # Plot SHAP summary bar
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, max_display=6, show=False)
    st.pyplot(fig)
