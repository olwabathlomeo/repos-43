import streamlit as st
import pandas as pd
import pickle
import shap

# Load the model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# SHAP init
shap.initjs()

st.title("🏦 Loan Approval Predictor")

# Input form
st.sidebar.header("Applicant Information")
no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, step=1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=0.0)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
res_assets = st.sidebar.number_input("Residential Assets Value", min_value=0.0)
comm_assets = st.sidebar.number_input("Commercial Assets Value", min_value=0.0)
bank_assets = st.sidebar.number_input("Bank Asset Value", min_value=0.0)

# Map categorical to numeric
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Form input
input_data = pd.DataFrame([[
    no_of_dependents, education, self_employed, loan_amount,
    loan_term, cibil_score, res_assets, comm_assets, bank_assets
]], columns=[
    'no_of_dependents', 'education', 'self_employed', 'loan_amount',
    'loan_term', 'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'bank_asset_value'
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success("✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected")

    # SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("Feature Impact (SHAP)")
    st_shap = shap.plots.waterfall(shap_values[0], max_display=9, show=False)
    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
