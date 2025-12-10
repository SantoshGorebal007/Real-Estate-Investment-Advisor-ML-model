# src/app/streamlit_app.py
# Single-entry Streamlit app (clean, robust)
# - ensures src/ is on sys.path
# - downloads models from Google Drive (via IDs in env / Streamlit secrets)
# - loads ProductionModel and provides three pages

import os
import sys
from pathlib import Path

# --- Ensure project root and src/ are on sys.path (required for Streamlit) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional debug prints if you set env var STREAMLIT_DEBUG=1
if os.getenv("STREAMLIT_DEBUG") == "1":
    print("DEBUG: PROJECT_ROOT=", PROJECT_ROOT)
    print("DEBUG: sys.path[0:4]=", sys.path[0:4])
    print("DEBUG: cwd=", os.getcwd())

import streamlit as st
import pandas as pd
import numpy as np
import json

# --- Google Drive downloader integration ---
DOWNLOAD_STATUS = None
DRIVE_CLASSIFIER_ID = os.getenv("DRIVE_CLASSIFIER_ID")
DRIVE_REGRESSOR_ID = os.getenv("DRIVE_REGRESSOR_ID")
DRIVE_FEATURE_LIST_ID = os.getenv("DRIVE_FEATURE_LIST_ID")

if DRIVE_CLASSIFIER_ID and DRIVE_REGRESSOR_ID and DRIVE_FEATURE_LIST_ID:
    try:
        # lazy import to avoid import-time errors if file missing
        from src.app.download_models import ensure_all_models_present
        ensure_all_models_present(DRIVE_CLASSIFIER_ID, DRIVE_REGRESSOR_ID, DRIVE_FEATURE_LIST_ID)
        DOWNLOAD_STATUS = "Models downloaded from Google Drive."
    except Exception as e:
        DOWNLOAD_STATUS = f"Model download failed: {e}"
else:
    DOWNLOAD_STATUS = "Drive IDs not set. Set DRIVE_CLASSIFIER_ID, DRIVE_REGRESSOR_ID, DRIVE_FEATURE_LIST_ID in Streamlit secrets or env."

# show download status
st.sidebar.info(DOWNLOAD_STATUS)

# --- Import ProductionModel AFTER downloader attempt (so models exist on disk) ---
try:
    from src.models.inference import ProductionModel
    prod_import_error = None
except Exception as e:
    ProductionModel = None
    prod_import_error = str(e)

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

PAGES = ["🏠 Property Prediction", "📊 Insights Dashboard", "🧠 Model Info"]
choice = st.sidebar.radio("Navigate", PAGES)

# Try to load production model, show friendly message if fails
pm = None
model_load_error = None
if ProductionModel is not None:
    try:
        pm = ProductionModel()
    except Exception as e:
        pm = None
        model_load_error = str(e)
else:
    model_load_error = prod_import_error or "ProductionModel class not found."

if model_load_error:
    st.sidebar.error("Model load issue: " + model_load_error)

##############################################
# PAGE 1 — PROPERTY PREDICTION
##############################################
if choice == "🏠 Property Prediction":
    st.title("🏠 Real Estate Investment Advisor — Property Prediction")
    st.write("Fill in property details; the app will predict Investment (Yes/No) and Future Price (5 years).")

    col1, col2 = st.columns(2)
    with col1:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
        sqft = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1200)
        price_lakhs = st.number_input("Current Price (Lakhs)", min_value=0.0, value=80.0, step=1.0)
        locality = st.text_input("Locality (optional)")

    with col2:
        floor_no = st.number_input("Floor No.", min_value=0, max_value=100, value=2)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=10)
        city = st.text_input("City (optional)")
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)

    # compute price_per_sqft
    price_per_sqft = (price_lakhs * 100000.0) / sqft if sqft > 0 else 0.0

    raw_input = {
        "BHK": bhk,
        "Size_in_SqFt": sqft,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": price_per_sqft,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Year_Built": year_built,
        "Locality": locality,
        "City": city
    }

    if st.button("Predict"):
        if pm is None:
            st.error("Model not loaded. Check sidebar for details.")
        else:
            try:
                with st.spinner("Predicting..."):
                    output = pm.predict_all(raw_input)
            except Exception as e:
                st.exception(f"Prediction failed: {e}")
                output = {"good_investment": None, "confidence": None, "future_price_5y": None}

            st.subheader("Results")
            gi = output.get("good_investment")
            conf = output.get("confidence")
            fut = output.get("future_price_5y")

            if gi is None:
                st.warning("Prediction returned empty result.")
            else:
                if gi == 1:
                    st.success("✅ Good Investment")
                else:
                    st.error("❌ Not a Good Investment")

            if fut is not None:
                st.metric("Predicted Price after 5 years (Lakhs)", f"{fut:.2f}")

            if conf is not None:
                st.write(f"Model confidence: **{conf*100:.1f}%**")

##############################################
# PAGE 2 — INSIGHTS DASHBOARD
##############################################
elif choice == "📊 Insights Dashboard":
    st.title("📊 Insights Dashboard")
    p = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
    if not p.exists():
        st.error(f"Processed dataset not found at {p}")
    else:
        df = pd.read_csv(p)
        st.sidebar.subheader("Filters")
        if "City" in df.columns:
            sel_city = st.sidebar.selectbox("City", sorted(df["City"].dropna().unique()))
            df_city = df[df["City"] == sel_city]
        else:
            st.sidebar.info("City column missing in processed data; using full dataset.")
            sel_city = None
            df_city = df.copy()

        st.subheader("Price per SqFt distribution")
        try:
            import plotly.express as px
            fig = px.histogram(df_city, x="Price_per_SqFt", nbins=40, title="Price per SqFt")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write("Plotly not installed or plotting failed. Showing simple stats.")
            st.write(df_city["Price_per_SqFt"].describe())

        st.subheader("Correlation (numeric features)")
        corr = df_city.select_dtypes(include=["number"]).corr()
        st.dataframe(corr.round(3))

        st.subheader("Top Feature Importances (if available)")
        fi_path = PROJECT_ROOT / "models" / "inspection" / "permutation_importance_top30.csv"
        if fi_path.exists():
            fi = pd.read_csv(fi_path)
            st.bar_chart(fi.set_index("feature")["perm_mean"].head(20))
        else:
            st.info("Permutation importance file not found. Run inspection script to create it.")

##############################################
# PAGE 3 — MODEL INFO
##############################################
elif choice == "🧠 Model Info":
    st.title("🧠 Model Info & MLflow")

    st.subheader("Model loading status")
    if pm is None:
        st.error("ProductionModel not loaded. Sidebar shows load error.")
        if model_load_error:
            st.code(model_load_error)
    else:
        st.success("ProductionModel loaded.")
        try:
            flen = len(pm.feature_list)
        except Exception:
            flen = "unknown"
        st.write(f"Production features: **{flen}**")
        try:
            st.write(pm.feature_list[:30])
        except Exception:
            st.write("Feature list not available for preview.")

    st.subheader("MLflow & Metrics")
    st.write("If you run `mlflow ui` locally, you can view detailed run artifacts.")
    cls_metrics = PROJECT_ROOT / "models" / "classification_clean" / "metrics.json"
    reg_metrics = PROJECT_ROOT / "models" / "regression_clean" / "metrics.json"
    if cls_metrics.exists():
        st.write("Classification metrics:")
        st.json(json.loads(cls_metrics.read_text()))
    else:
        st.info("Classification metrics.json not found (check models/classification_clean/).")

    if reg_metrics.exists():
        st.write("Regression metrics:")
        st.json(json.loads(reg_metrics.read_text()))
    else:
        st.info("Regression metrics.json not found (check models/regression_clean/).")
