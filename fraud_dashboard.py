import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ------------------- Page Config -------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ------------------- Custom CSS Styling -------------------
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1605902711622-cfb43c4437d1');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        backdrop-filter: blur(10px);
    }
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        text-align: center;
        color: white;
    }
    .hero-title {
        font-size: 3.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }
    .hero-text {
        font-size: 1.4em;
        max-width: 800px;
        margin: 0 auto;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .start-button {
        margin-top: 2em;
        font-size: 1.2em;
        padding: 0.75em 2em;
        border: none;
        border-radius: 50px;
        background: linear-gradient(135deg, #00feba, #5b5bff);
        color: white;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 0 20px #00feba88;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px #00feba88; }
        50% { box-shadow: 0 0 30px #5b5bffaa; }
        100% { box-shadow: 0 0 10px #00feba88; }
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Hero Section -------------------
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">💳 Credit Card Fraud Detection</div>
        <div class="hero-text">
            Welcome! This app simulates, monitors, and detects fraudulent financial transactions using machine learning.
        </div>
        <button class="start-button" onclick="window.scrollTo({ top: 800, behavior: 'smooth' });">🚀 Start Simulation</button>
    </div>
""", unsafe_allow_html=True)

# ------------------- Load Model & Dataset -------------------
model = joblib.load("fraud_model.pkl")
df_real = pd.read_csv("creditcard.csv")

# ------------------- Manual Fraud Prediction -------------------
st.header("🔍 Manual Fraud Prediction")
with st.expander("Input 30 Features to Predict Fraud"):
    cols = st.columns(3)
    input_data = []
    for i in range(30):
        with cols[i % 3]:
            val = st.number_input(f"Feature {i+1}", format="%.5f", key=f"feature_{i+1}")
            input_data.append(val)

    if st.button("Predict Fraud"):
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        features_df = pd.DataFrame([input_data], columns=columns)
        prediction = model.predict(features_df)[0]

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

# ------------------- Simulated Transaction Generator -------------------
def simulate_from_real_data(df, n=100, fraud_ratio=0.15):
    fraud_df = df[df['Class'] == 1]
    legit_df = df[df['Class'] == 0]

    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    fraud_samples = fraud_df.sample(n=n_fraud)
    legit_samples = legit_df.sample(n=n_legit)

    combined_df = pd.concat([fraud_samples, legit_samples]).sample(frac=1).reset_index(drop=True)

    data = []
    for _, row in combined_df.iterrows():
        is_fraud = int(row['Class'])
        data.append({
            "customerName": f"User{random.randint(1000, 9999)}",
            "merchantName": f"Shop{random.randint(100, 999)}",
            "amount": round(row['Amount'], 2),
            "fraud": is_fraud,
            "fraudLevel": "high" if is_fraud == 1 else "low",
            "timestamp": pd.Timestamp.now() - pd.to_timedelta(random.randint(0, 3600), unit="s")
        })

    return pd.DataFrame(data)

df = simulate_from_real_data(df_real, 100)

# ------------------- Fraud Overview Metrics -------------------
st.header("📊 Fraud Overview")
fraud_count = df["fraud"].sum()
legit_count = len(df) - fraud_count
fraud_pct = round(fraud_count / len(df) * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("Fraudulent Transactions", fraud_count, f"{fraud_pct}%")
col2.metric("Legitimate Transactions", legit_count)
col3.metric("Total Volume ($)", f"${df['amount'].sum():,.2f}")

# ------------------- Transaction Analytics -------------------
st.subheader("📈 Transaction Analytics")
c1, c2 = st.columns(2)

with c1:
    pie = px.pie(df, names="fraudLevel", title="Fraud Distribution")
    st.plotly_chart(pie, use_container_width=True)

with c2:
    df_sorted = df.sort_values("timestamp")
    line = px.line(df_sorted, x="timestamp", y="amount", title="Transaction Volume Over Time")
    st.plotly_chart(line, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    risky_merchants = df[df["fraud"] == 1]["merchantName"].value_counts().reset_index()
    risky_merchants.columns = ["merchant", "fraud_count"]
    if not risky_merchants.empty:
        bar = px.bar(risky_merchants, x="merchant", y="fraud_count", title="Top High-Risk Merchants")
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("No high-risk merchants detected yet.")

with c4:
    st.subheader("🧾 Recent Transactions")
    st.dataframe(df[["customerName", "merchantName", "amount", "fraudLevel", "timestamp"]].head(10), use_container_width=True)

# ------------------- Deep EDA -------------------
st.header("🔬 Deep Data Visualization & EDA")
eda_df = df_real.copy()

with st.expander("🔍 Explore the Raw Data"):
    st.dataframe(eda_df.sample(100), use_container_width=True)

st.subheader("Class Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="Class", data=eda_df, palette="Set2", ax=ax1)
ax1.set_xticklabels(["Legit", "Fraud"])
st.pyplot(fig1)

st.subheader("Transaction Amount Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(eda_df["Amount"], bins=50, kde=True, ax=ax2)
st.pyplot(fig2)

st.subheader("Time vs Amount")
fig3, ax3 = plt.subplots()
sns.scatterplot(x="Time", y="Amount", hue="Class", data=eda_df.sample(1000), palette="coolwarm", alpha=0.6, ax=ax3)
st.pyplot(fig3)

st.subheader("Feature Correlation Heatmap")
corr = eda_df.drop("Class", axis=1).corr()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="Blues", ax=ax4)
st.pyplot(fig4)

st.subheader("Top Correlated Features with Class")
top_corr_features = eda_df.corr()["Class"].abs().sort_values(ascending=False)[1:6].index.tolist()
fig5, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()
for i, feat in enumerate(top_corr_features):
    sns.boxplot(x="Class", y=feat, data=eda_df, ax=axs[i], palette="Set3")
    axs[i].set_title(feat)
st.pyplot(fig5)

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Rajsv Mahendra 2025")
