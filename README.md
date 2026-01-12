# credit-card-fraud-detection-ml
Production-ready credit card fraud detection system using XGBoost and machine learning pipelines on large datasets.


# 💳 Credit Card Fraud Detection System

An end-to-end **Machine Learning–based Credit Card Fraud Detection system** built using **Python, XGBoost, and Scikit-learn**, designed for real-world, large-scale datasets. The project includes proper preprocessing, imbalance handling, model evaluation, and a production-ready saved pipeline for deployment.

---

## 🚀 Project Overview

Credit card fraud is a critical problem in financial systems where fraudulent transactions are extremely rare compared to normal ones. This project addresses that challenge by:

* Handling **large, imbalanced datasets**
* Applying **robust preprocessing pipelines**
* Training an **XGBoost classifier** optimized for fraud detection
* Saving the trained model using **pickle/joblib** for production use

The model predicts whether a transaction is **fraudulent or legitimate**, along with fraud probability.

---

## 🧠 Machine Learning Pipeline

The project uses a **Scikit-learn Pipeline** to ensure consistency between training and inference.

### Pipeline Components:

1. **Data Preprocessing**

   * Numerical feature scaling using `StandardScaler`
   * Categorical feature encoding using `OneHotEncoder`
   * Automatic handling of unseen categories

2. **Model**

   * `XGBoost Classifier (XGBClassifier)`
   * Handles class imbalance using `scale_pos_weight`

3. **Evaluation Metrics**

   * Precision, Recall, F1-score
   * ROC-AUC score (preferred for imbalanced data)

---

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── credit_card_dataset.csv
│
├── models/
│   └── fraud_pipeline.pkl
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

* **Type:** Real-world / large-scale transactional dataset
* **Target Column:** `IsFraud`

  * `0` → Normal Transaction
  * `1` → Fraudulent Transaction
* **Features Include:**

  * Transaction Amount
  * Merchant ID
  * Transaction Type
  * Location
  * Other transactional attributes

> ⚠️ Dataset is not uploaded due to size and privacy constraints.

---

## ⚙️ Model Training

* Train-test split with stratification
* Class imbalance handled using weighted loss
* Hyperparameters tuned for fraud detection

The trained model is saved as a **single pipeline file**:

```
models/fraud_pipeline.pkl
```

This file includes both preprocessing and the trained model.

---

## 🔮 Prediction

The saved pipeline can be directly used for inference:

* No manual preprocessing required
* Accepts raw transaction data
* Outputs:

  * Fraud / Normal classification
  * Fraud probability score

---

## 🖥️ Deployment

The project is deployment-ready and can be integrated with:

* **Streamlit** (Interactive Web App)
* **Cloud Platforms** (Render, Streamlit Cloud, AWS, etc.)

---

## 📌 Key Highlights

* End-to-end ML pipeline
* Large dataset handling
* Imbalance-aware fraud detection
* Production-ready model serialization
* Clean, modular project structure

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Joblib / Pickle
* Streamlit (for UI)

---

## 📈 Future Improvements

* SHAP-based model explainability
* Threshold tuning for business optimization
* Real-time transaction streaming
* Model monitoring and retraining

---

## 👨‍💻 Author

**Vaibhav Singh**
Aspiring NLP & ML Engineer | Data Science Student

---

⭐ If you find this project useful, feel free to star the repository!

