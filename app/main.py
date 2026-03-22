import sys
import os
from pathlib import Path

# Fix for Streamlit Cloud Pathing
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
import joblib
from src.data_processor import load_and_merge_data
from src.model import train_model # Make sure this matches your filename

st.set_page_config(page_title="Health Care Equity", layout="wide")

try:
    # 1. Load Data
    df, report = load_and_merge_data()

    # 2. Train Model (Only if pkl doesn't exist or needs update)
    model_path = os.path.join(root_path, "models", "cost_predictor.pkl")
    
    # This cleans and trains safely
    if not os.path.exists(model_path):
        with st.spinner("Initializing Predictive Engine..."):
            train_model(df)

    # 3. Load the Model
    model = joblib.load(model_path)

    st.title("🏥 Health Equity Insights Dashboard")
    
    # --- DASHBOARD UI ---
    st.write("Data successfully loaded and model trained.")
    st.dataframe(report.head())

except Exception as e:
    st.error("Dashboard Initialization Error")
    st.exception(e)
