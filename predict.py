# ===============================
# predict.py
# Credit Card Fraud Detection
# ===============================

import pandas as pd
import numpy as np
import joblib


# -------------------------------
# STEP 1: Load Trained Pipeline
# -------------------------------
MODEL_PATH = "fraud_detection_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)
print("✅ Model pipeline loaded successfully")


# -------------------------------
# STEP 2: Create Sample Input
# (same columns as training X)
# -------------------------------

sample_transaction = {
    "Amount": 4189.27,
    "MerchantID": 688,
    "TransactionType": "refund",
    "Location": "San Antonio",
    "TransactionDate": "15:35.5"   # will be dropped internally
}

df_new = pd.DataFrame([sample_transaction])


# -------------------------------
# STEP 3: Same Feature Engineering
# -------------------------------

# Log transform Amount
df_new["Amount_log"] = np.log1p(df_new["Amount"])
df_new.drop(columns=["Amount"], inplace=True)

# Drop TransactionDate (same as training)
df_new.drop(columns=["TransactionDate"], inplace=True)


# -------------------------------
# STEP 4: Prediction
# -------------------------------

prediction = pipeline.predict(df_new)[0]
probability = pipeline.predict_proba(df_new)[0][1]


# -------------------------------
# STEP 5: Human-Readable Output
# -------------------------------

if prediction == 1:
    print("❌ FRAUD TRANSACTION DETECTED")
else:
    print("✅ NORMAL TRANSACTION")

print(f"Fraud Probability: {probability:.4f}")