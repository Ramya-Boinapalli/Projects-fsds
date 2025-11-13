# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:25:47 2025

@author: Ramya
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# --- Load Trained Model ---
model_path = 'models/rf_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

# --- Streamlit App Layout ---
st.title("🌍 CO₂ Concentration Prediction Dashboard")
st.markdown("""
This app predicts **atmospheric CO₂ concentration (ppm)** using a trained Machine Learning model.  
You can make **single predictions** or **batch predictions** by uploading a CSV.
""")

col1, col2 = st.columns(2)

# --- Single Prediction ---
with col1:
    st.header("📈 Single Prediction")
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2020)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    decimal_date = st.number_input("Decimal Date", min_value=1950.0, max_value=2100.0, value=2020.0, format="%.3f")

    if st.button("🔍 Predict CO₂ (ppm)"):
        if model is None:
            st.error("❌ Model not found! Please run train_model.py first to generate rf_model.pkl.")
        else:
            X_in = np.array([[year, month, decimal_date]])
            pred = model.predict(X_in)[0]
            st.success(f"✅ Predicted CO₂ Concentration: **{pred:.2f} ppm**")

# --- Batch Prediction ---
with col2:
    st.header("🗂️ Batch Predictions")
    uploaded = st.file_uploader("Upload CSV with columns: year, month, decimal_date", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("📋 Preview of Uploaded Data:", df.head())

        required_cols = ["year", "month", "decimal_date"]
        if not all(c in df.columns for c in required_cols):
            st.error(f"❌ CSV must contain columns: {required_cols}")
        else:
            if model is None:
                st.error("Model not found. Run training first.")
            else:
                preds = model.predict(df[required_cols])
                df["predicted_co2_ppm"] = preds
                st.success("✅ Batch prediction complete!")
                st.dataframe(df.head(20))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions CSV",
                    data=csv,
                    file_name="co2_predictions.csv",
                    mime="text/csv"
                )

# --- Visualization Section ---
st.markdown("---")
st.header("📊 Exploratory Data Visualizations")

if os.path.exists("data/co2_emissions.csv"):
    df_full = pd.read_csv("data/co2_emissions.csv")
    if "co2_ppm" in df_full.columns:
        st.subheader("Average CO₂ Levels by Year")
        yearly_avg = df_full.groupby("year")["co2_ppm"].mean().reset_index()

        fig, ax = plt.subplots()
        ax.plot(yearly_avg["year"], yearly_avg["co2_ppm"], color="teal", linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ (ppm)")
        ax.set_title("Yearly Average CO₂ Concentration")
        st.pyplot(fig)

        st.subheader("Feature Correlation Matrix")
        numeric = df_full.select_dtypes(include=[np.number])
        corr = numeric.corr()
        st.dataframe(corr)
    else:
        st.warning("⚠️ The dataset in data/co2_emissions.csv doesn’t contain 'co2_ppm' column.")
else:
    st.info("ℹ️ Place your dataset at 'data/co2_emissions.csv' to view example visuals.")

st.markdown("---")
st.caption("🧠 Built with Streamlit + Random Forest | Dataset: Atmospheric CO₂ ppm Data")
