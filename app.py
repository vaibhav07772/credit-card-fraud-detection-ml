# ===============================
# app.py
# Streamlit Fraud Detection App
# ===============================

import streamlit as st

# 🔥 MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

import pandas as pd
import numpy as np
import joblib


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()


# -------------------------------
# App UI
# -------------------------------
st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app uses a **Machine Learning pipeline (XGBoost)** to detect "
    "**fraudulent transactions** in real-world scenarios."
)

st.divider()

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔍 Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

merchant_id = st.number_input(
    "Merchant ID",
    min_value=1,
    value=100
)

transaction_type = st.selectbox(
    "Transaction Type",
    ["purchase", "refund"]
)

location = st.selectbox(
    "Location",
    [
        "New York", "Dallas", "San Antonio",
        "Philadelphia", "Chicago", "San Jose",
        "Phoenix", "Houston", "San Diego"
    ]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Fraud"):

    # ✅ Step 1: Create input
    input_data = pd.DataFrame([{
        "Amount": amount,
        "MerchantID": merchant_id,
        "TransactionType": transaction_type,
        "Location": location,
        "TransactionDate": "00:00.0"
    }])

    # ✅ Step 2: Feature engineering (INSIDE button)
    input_data["Amount_log"] = np.log1p(input_data["Amount"])
    input_data.drop(columns=["Amount", "TransactionDate"], inplace=True)

    # ✅ Step 3: Prediction
    prob = model.predict_proba(input_data)[0][1]

    THRESHOLD = 0.25  # fraud threshold

    if prob >= THRESHOLD:
        st.error("❌ FRAUD TRANSACTION DETECTED")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.write(f"**Fraud Probability:** `{prob:.4f}`")
    st.caption(f"Threshold used: {THRESHOLD}")

st.divider()
st.caption("👨‍💻 Built by Vaibhav Singh | ML Engineer Project")