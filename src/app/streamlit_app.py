# src/app/streamlit_app.py
# Enhanced Streamlit UI for Real Estate Investment Advisor
# - keeps downloader & inference behavior
# - adds improved UI components: sliders, result card, confidence gauge,
#   city bar chart, investment pie chart, quick examples, feature importance preview

import os
import sys
from pathlib import Path

# Ensure project root / src are on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import json

# ---------- Google Drive downloader (if used) ----------
DOWNLOAD_STATUS = None
DRIVE_CLASSIFIER_ID = os.getenv("DRIVE_CLASSIFIER_ID")
DRIVE_REGRESSOR_ID = os.getenv("DRIVE_REGRESSOR_ID")
DRIVE_FEATURE_LIST_ID = os.getenv("DRIVE_FEATURE_LIST_ID")

if DRIVE_CLASSIFIER_ID and DRIVE_REGRESSOR_ID and DRIVE_FEATURE_LIST_ID:
    try:
        from src.app.download_models import ensure_all_models_present
        ensure_all_models_present(DRIVE_CLASSIFIER_ID, DRIVE_REGRESSOR_ID, DRIVE_FEATURE_LIST_ID)
        DOWNLOAD_STATUS = "Models downloaded from Google Drive."
    except Exception as e:
        DOWNLOAD_STATUS = f"Model download failed: {e}"
else:
    DOWNLOAD_STATUS = "Drive IDs not set. Set DRIVE_CLASSIFIER_ID, DRIVE_REGRESSOR_ID, DRIVE_FEATURE_LIST_ID in Streamlit secrets or env."

# show download status in sidebar
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
st.sidebar.info(DOWNLOAD_STATUS)

# ---------- Import production model after downloader ----------
try:
    from src.models.inference import ProductionModel
    prod_import_error = None
except Exception as e:
    ProductionModel = None
    prod_import_error = str(e)

# ---------- App layout / navigation ----------
PAGES = ["🏠 Property Prediction", "📊 Insights Dashboard", "🧠 Model Info"]
choice = st.sidebar.radio("Navigate", PAGES)

# UX: Quick example presets
EXAMPLES = {
    "Default: 2BHK, 1200 sqft, 80L": {"BHK": 2, "Size_in_SqFt": 1200, "Price_in_Lakhs": 80},
    "Family: 3BHK, 1500 sqft, 120L": {"BHK": 3, "Size_in_SqFt": 1500, "Price_in_Lakhs": 120},
    "Luxury: 4BHK, 3000 sqft, 400L": {"BHK": 4, "Size_in_SqFt": 3000, "Price_in_Lakhs": 400},
}

# Load model if possible
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

# ---------- Helpers ----------
def compute_price_per_sqft(price_lakhs, size_sqft):
    return (price_lakhs * 100000.0) / size_sqft if size_sqft and size_sqft > 0 else 0.0

def render_prediction_card(gi, confidence, future_price):
    """Render a nicer result card with color-coded status and a confidence gauge."""
    col1, col2 = st.columns([2, 1])
    with col1:
        if gi == 1:
            st.markdown("<div style='background:#e6fff0;padding:14px;border-radius:8px'>", unsafe_allow_html=True)
            st.markdown("### ✅ Good Investment", unsafe_allow_html=True)
        elif gi == 0:
            st.markdown("<div style='background:#fff0f0;padding:14px;border-radius:8px'>", unsafe_allow_html=True)
            st.markdown("### ❌ Not a Good Investment", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#f0f8ff;padding:14px;border-radius:8px'>", unsafe_allow_html=True)
            st.markdown("### ⚠️ Prediction unavailable", unsafe_allow_html=True)

        # Key metrics
        if future_price is not None:
            st.metric("Predicted Price after 5 years (Lakhs)", f"{future_price:.2f}")
        else:
            st.write("Predicted Price after 5 years: —")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.write("**Confidence**")
        if confidence is None:
            st.write("N/A")
        else:
            pct = float(confidence) * 100.0
            st.progress(min(max(pct / 100.0, 0.0), 1.0))
            st.write(f"{pct:.1f}%")
        st.write("")  # spacing
        # short interpretation
        if gi == 1:
            st.info("Model suggests this property is likely to appreciate relative to peers.")
        elif gi == 0:
            st.warning("Model suggests limited upside; consider negotiation or alternative properties.")
        else:
            st.write("No interpretation available.")

def safe_plotly_bar(df, x, y, title, height=350):
    try:
        import plotly.express as px
        fig = px.bar(df, x=x, y=y, title=title, height=height)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write(f"[Plotly not available] {title}")
        st.dataframe(df[[x, y]])

def safe_plotly_pie(labels, values, title, height=300):
    try:
        import plotly.express as px
        fig = px.pie(names=labels, values=values, title=title, height=height)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write(f"[Plotly not available] {title}")
        st.write(pd.DataFrame({"label": labels, "value": values}))

# ---------- Page: Property Prediction ----------
if choice == "🏠 Property Prediction":
    st.title("🏠 Real Estate Investment Advisor — Property Prediction")
    st.write("Fill in property details and get investment recommendation + 5-year forecast.")

    # Left column: inputs
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Property Inputs")
        # Quick examples
        example = st.selectbox("Quick Example", ["Custom"] + list(EXAMPLES.keys()))
        if example != "Custom":
            ex = EXAMPLES[example]
            default_bhk = ex["BHK"]
            default_size = ex["Size_in_SqFt"]
            default_price = ex["Price_in_Lakhs"]
        else:
            default_bhk = 2
            default_size = 1200
            default_price = 80

        bhk = st.slider("BHK", 1, 6, value=default_bhk)
        sqft = st.slider("Size (SqFt)", 200, 5000, value=default_size, step=50)
        price_lakhs = st.slider("Current Price (Lakhs)", 1.0, 2000.0, value=float(default_price), step=1.0)
        locality = st.text_input("Locality (optional)")
        city = st.text_input("City (optional)")
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)

        # Advanced toggles
        with st.expander("Advanced inputs (optional)"):
            floor_no = st.number_input("Floor No.", min_value=0, max_value=100, value=2)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=10)
            furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi", "Fully"])
    # Right column: quick info + help
    with right:
        st.subheader("Quick Info")
        st.write("Use sliders to rapidly test scenarios or select a Quick Example.")
        st.write("Tip: Use the Insights page to inspect city-level price trends.")
        st.markdown("---")
        st.write("Input summary:")
        st.write(f"BHK: **{bhk}**, Size: **{sqft} sqft**, Price: **{price_lakhs} Lakhs**")

    # compute derived features
    price_per_sqft = compute_price_per_sqft(price_lakhs, sqft)

    # Prepare input dict for model
    user_input = {
        "BHK": bhk,
        "Size_in_SqFt": sqft,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": price_per_sqft,
        "Floor_No": locals().get("floor_no", 0),
        "Total_Floors": locals().get("total_floors", 0),
        "Year_Built": year_built,
        "Locality": locality or "",
        "City": city or ""
    }

    # Predict button
    if st.button("Predict"):
        if pm is None:
            st.error("Model not loaded. Check sidebar for details.")
        else:
            try:
                with st.spinner("Running prediction..."):
                    out = pm.predict_all(user_input)
            except Exception as e:
                st.exception(f"Prediction failed: {e}")
                out = {"good_investment": None, "confidence": None, "future_price_5y": None}

            gi = out.get("good_investment")
            conf = out.get("confidence")
            fut = out.get("future_price_5y")

            render_prediction_card(gi, conf, fut)

            # Show small feature importance if available (per model inspection file)
            fi_path = PROJECT_ROOT / "models" / "inspection" / "permutation_importance_top30.csv"
            if fi_path.exists():
                try:
                    fi = pd.read_csv(fi_path).head(10)
                    st.subheader("Top features (local view)")
                    safe_plotly_bar(fi, "feature", "perm_mean", "Top Permutation Importances (top 10)", height=260)
                except Exception:
                    pass

# ---------- Page: Insights Dashboard ----------
elif choice == "📊 Insights Dashboard":
    st.title("📊 Insights Dashboard")
    p = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
    if not p.exists():
        st.error(f"Processed dataset not found at {p}")
    else:
        df = pd.read_csv(p)
        st.sidebar.subheader("Filters")
        city_list = sorted(df["City"].dropna().unique()) if "City" in df.columns else []
        sel_city = st.sidebar.selectbox("City", ["All"] + city_list)

        # Show aggregate stats
        st.subheader("Market snapshot")
        num_props = int(len(df))
        avg_price = df["Price_in_Lakhs"].mean() if "Price_in_Lakhs" in df.columns else np.nan
        avg_pps = df["Price_per_SqFt"].mean() if "Price_per_SqFt" in df.columns else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Properties (total)", f"{num_props:,}")
        c2.metric("Avg Price (Lakhs)", f"{avg_price:.2f}" if not np.isnan(avg_price) else "N/A")
        c3.metric("Avg Price/SqFt", f"{avg_pps:.2f}" if not np.isnan(avg_pps) else "N/A")

        st.markdown("---")
        st.subheader("Price per SqFt by City")
        if "Price_per_SqFt" in df.columns and "City" in df.columns:
            city_pps = df.groupby("City")["Price_per_SqFt"].median().reset_index().sort_values("Price_per_SqFt", ascending=False)
            safe_plotly_bar(city_pps.head(25), "City", "Price_per_SqFt", "Top 25 Cities by Median Price/SqFt", height=500)
        else:
            st.info("Price_per_SqFt or City columns missing.")

        st.markdown("---")
        st.subheader("Investment ratio (Good vs Not Good)")
        # If Good_Investment exists in processed data, show pie chart; else attempt to compute
        if "Good_Investment" in df.columns:
            counts = df["Good_Investment"].value_counts().sort_index()
            labels = ["Not Good", "Good"] if 0 in counts.index or 1 in counts.index else list(counts.index.astype(str))
            values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
            safe_plotly_pie(labels, values, "Investment ratio across dataset")
        else:
            st.info("Good_Investment label not present in processed data.")

# ---------- Page: Model Info ----------
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
            flist = pm.feature_list
            st.write(f"Production features: **{len(flist)}**")
            st.download_button("Download feature list (CSV)", "\n".join(flist), file_name="used_feature_list.txt")
            st.write(flist[:60])
        except Exception:
            st.write("Feature list not available.")

    st.markdown("---")
    st.subheader("Model metrics (snapshot)")
    cls_metrics = PROJECT_ROOT / "models" / "classification_clean" / "metrics.json"
    reg_metrics = PROJECT_ROOT / "models" / "regression_clean" / "metrics.json"
    if cls_metrics.exists():
        st.write("Classification metrics:")
        try:
            st.json(json.loads(cls_metrics.read_text()))
        except Exception:
            st.write(cls_metrics.read_text())
    else:
        st.info("Classification metrics.json not found.")

    if reg_metrics.exists():
        st.write("Regression metrics:")
        try:
            st.json(json.loads(reg_metrics.read_text()))
        except Exception:
            st.write(reg_metrics.read_text())
    else:
        st.info("Regression metrics.json not found.")

    st.markdown("---")
    st.subheader("Top feature importances (if available)")
    fi_path = PROJECT_ROOT / "models" / "inspection" / "permutation_importance_top30.csv"
    if fi_path.exists():
        try:
            fi = pd.read_csv(fi_path)
            safe_plotly_bar(fi.head(30), "feature", "perm_mean", "Permutation Importances (top 30)", height=420)
        except Exception:
            st.write("Could not render feature importance.")
    else:
        st.info("Permutation importance file not found. Run inspection script to create it.")
