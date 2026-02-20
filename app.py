import streamlit as st
import pandas as pd
import joblib

fraud_model = joblib.load("fraud_model_detection.pkl")
model = fraud_model["model"]
threshold = fraud_model["threshold"]

st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and click the Predict button")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT","DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if transaction_type == "DEPOSIT":
    transaction_type_for_model = "DEBIT"
else:
    transaction_type_for_model = transaction_type

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type_for_model,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Engineer same features as training
    input_data["balanceDiffOrig"] = input_data["oldbalanceOrg"] - input_data["newbalanceOrig"]
    input_data["balanceDiffDest"] = input_data["newbalanceDest"] - input_data["oldbalanceDest"]

    probability = model.predict_proba(input_data)[:, 1][0]

    st.write("Fraud Probability:", f"{probability*100:.2f}%")

    if probability >= threshold:
        st.error(f"🔴 FRAUD DETECTED (Probability: {probability:.2f})")
    elif probability >= 0.75:
        st.warning(f"🟠 HIGH RISK TRANSACTION (Probability: {probability:.2f})")
    else:
        st.success("✅ Transaction is likely legitimate.")