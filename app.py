import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import pickle
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# Load the trained XGBoost model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant details.")

# Sidebar for input features
st.sidebar.header("Enter Applicant Details")

def user_input():
    no_of_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 1)
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.sidebar.number_input("Annual Income (KES)", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount (KES)", min_value=0)
    loan_term = st.sidebar.slider("Loan Term (months)", 12, 360, 120)
    cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 700)
    residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0)
    luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0)
    bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0)

    data = {
        "no_of_dependents": no_of_dependents,
        "education": 1 if education == "Graduate" else 0,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("Applicant Information")
st.write(input_df)

# Predict
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("Prediction")
st.write("✅ **Loan Approved**" if prediction == 1 else "❌ **Loan Rejected**")
st.write(f"**Probability of Approval:** {prediction_proba[1]:.2%}")
st.write(f"**Probability of Rejection:** {prediction_proba[0]:.2%}")

# SHAP Explanation
st.subheader("🔍 Feature Impact (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# Display SHAP force plot
st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True), height=300)

# Optional: SHAP bar plot
st.subheader("Feature Importance (Bar)")
fig, ax = plt.subplots()
shap.plots.bar(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_df.iloc[0]), show=False)
st.pyplot(fig)
