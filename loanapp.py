# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn   # Needed for loading the pickle model

# -------------------------------------------------------
# Load trained model (.pkl)
# -------------------------------------------------------
with open("my_final_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------------------------------
# Streamlit App Title
# -------------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; background-color: #ffeecc; padding: 12px; color: #cc6600;'>"
    "<b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

st.subheader("Enter Applicant Details:")

# -------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------

# Numeric inputs
granted_loan = st.number_input("Granted Loan Amount", min_value=0, max_value=500000, step=500)
requested_loan = st.number_input("Requested Loan Amount", min_value=0, max_value=500000, step=500)
fico = st.slider("FICO Score", min_value=300, max_value=850, step=1)
income = st.number_input("Monthly Gross Income", min_value=0, max_value=50000, step=100)
housing = st.number_input("Monthly Housing Payment", min_value=0, max_value=20000, step=50)
bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", ["Yes", "No"])
bounty = st.number_input("Bounty", min_value=0, max_value=1000, step=1)

# Categorical inputs (from your CSV)
reason = st.selectbox("Reason for Loan", ["Home Improvement", "Debt Consolidation", "Other"])
fico_group = st.selectbox("FICO Score Group", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Retired"])
employment_sector = st.selectbox("Employment Sector", ["Private", "Government", "Nonprofit", "Other"])
lender = st.selectbox("Lender", ["Bank A", "Bank B", "Bank C", "Other"])

# -------------------------------------------------------
# Create DataFrame from Inputs
# -------------------------------------------------------
input_data = pd.DataFrame({
    "Reason": [reason],
    "Granted_Loan_Amount": [granted_loan],
    "Requested_Loan_Amount": [requested_loan],
    "FICO_score": [fico],
    "Fico_Score_group": [fico_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Monthly_Gross_Income": [income],
    "Monthly_Housing_Payment": [housing],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    "Lender": [lender],
    "bounty": [bounty]
})

# -------------------------------------------------------
# Preprocessing (One-Hot Encode + Align to Model Features)
# -------------------------------------------------------

# 1. One-hot encode categorical columns
categorical_cols = ["Reason", "Fico_Score_group", "Employment_Status",
                    "Employment_Sector", "Ever_Bankrupt_or_Foreclose", "Lender"]
input_encoded = pd.get_dummies(input_data, columns=categorical_cols)

# 2. Add missing columns the model expects
model_columns = model.feature_names_in_

for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# 3. Ensure correct order
input_encoded = input_encoded[model_columns]

# -------------------------------------------------------
# prediction
# -------------------------------------------------------
if st.button("Evaluate Loan"):
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("🚫 Loan Predicted: **NOT Approved**")
    else:
        st.success("✅ Loan Predicted: **APPROVED**")
