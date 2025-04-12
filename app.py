import streamlit as st
import numpy as np
import joblib
import pandas as pd


# Load the trained model
model = joblib.load("fraud_model.pkl")

# Title of the app
st.title("Credit Card Fraud Detection")

st.markdown("""
This tool takes in 30 anonymized features of a credit card transaction  
and predicts whether it's **fraudulent** or **legitimate**.  
All data should be normalized — just like in model training.
""")

cols = st.columns(3)
input_data = []

for i in range(30):
    with cols[i % 3]:
        val = st.number_input(f"Feature {i+1}", format="%.5f", key=f"feature_{i+1}")
        input_data.append(val)

if st.button("Predict"):
    # Define column names like your model saw during training
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Create a DataFrame with one row and proper column names
    features_df = pd.DataFrame([input_data], columns=columns)

    # Predict using the DataFrame
    prediction = model.predict(features_df)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

