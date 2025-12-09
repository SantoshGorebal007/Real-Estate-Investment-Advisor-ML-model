# src/models/train_regression_clean.py
"""
Leak-safe regression training script.
Drops columns that likely leak the target before training.
Also reads models/auto_dropped_features.txt to remove explicit columns.
Trains RandomForestRegressor (+ XGBoost if available), logs to MLflow,
saves best model to models/regression_clean/.
"""
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.models.model_evaluation import evaluate_regression

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Paths / constants
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
OUT_DIR = PROJECT_ROOT / "models" / "regression_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Future_Price_5Y"

# Patterns to detect likely leakage / target-derived features
LEAK_PATTERNS = [
    "good_investment", "future_price", "future", "multi_factor",
    "price_le_median", "pps_le_median", "_le_median", "best_",
    "label", "_target", "good_"
]

# Auto-drop file (exact names to remove)
AUTO_DROP_FILE = PROJECT_ROOT / "models" / "auto_dropped_features.txt"

def read_auto_drop():
    if AUTO_DROP_FILE.exists():
        try:
            txt = AUTO_DROP_FILE.read_text().strip().splitlines()
            txt = [s.strip() for s in txt if s.strip()]
            print("[INFO] Read auto-drop list:", txt)
            return set(txt)
        except Exception as e:
            print("[WARN] Could not read auto-drop file:", e)
            return set()
    return set()

def load_and_clean(path=DATA_PATH, target=TARGET):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {path}")

    # drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # find pattern-matched cols
    cols_to_drop = set()
    for c in df.columns:
        lc = c.lower()
        for p in LEAK_PATTERNS:
            if p in lc and c != target:
                cols_to_drop.add(c)

    # include explicit auto-drop names
    auto_drop = read_auto_drop()
    if auto_drop:
        print("[INFO] Adding auto-drop columns to removal list:", sorted(list(auto_drop)))
    cols_to_drop.update(auto_drop)

    print(f"[INFO] Columns detected as potential leakage (will be removed from features): {sorted(list(cols_to_drop))}")

    # prepare X and y
    y = df[target].copy()
    X = df.drop(columns=[target] + list(cols_to_drop), errors='ignore')

    # ensure numeric (drop objects if any remain)
    X_num = X.select_dtypes(include=[np.number])
    dropped_obj_cols = [c for c in X.columns if c not in X_num.columns]
    if dropped_obj_cols:
        print(f"[INFO] Dropping non-numeric columns (expected if OHE already done): {dropped_obj_cols}")

    return X_num, y

def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_xgb(X_train, y_train):
    if not HAVE_XGB:
        return None
    xgb = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    return xgb

def run():
    try:
        print("[INFO] Loading and cleaning data...")
        X, y = load_and_clean()
    except Exception as e:
        print("[FATAL] Failed to load/clean data:", e)
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model, best_name = None, None
    best_rmse = float("inf")

    mlflow.set_experiment("RealEstate_Regression_Clean")
    with mlflow.start_run(run_name="baseline_clean"):
        # Random Forest
        try:
            print("[INFO] Training RandomForestRegressor...")
            rf = train_rf(X_train, y_train)
            y_pred = rf.predict(X_test)
            metrics_rf = evaluate_regression(y_test, y_pred, prefix="rf_reg_clean")
            mlflow.log_metric("rf_rmse", metrics_rf.get("rmse", None))
            if metrics_rf.get("rmse", float("inf")) < best_rmse:
                best_rmse = metrics_rf.get("rmse")
                best_model = rf
                best_name = "random_forest"
            joblib.dump(rf, OUT_DIR / "random_forest_regressor.pkl")
            mlflow.sklearn.log_model(rf, "random_forest_regressor")
            print("[INFO] RandomForestRegressor saved & logged.")
        except Exception as e:
            print("[WARN] RF regressor failed:", e)

        # XGBoost
        try:
            if HAVE_XGB:
                print("[INFO] Training XGBoostRegressor...")
                xgb = train_xgb(X_train, y_train)
                y_pred_x = xgb.predict(X_test)
                metrics_x = evaluate_regression(y_test, y_pred_x, prefix="xgb_reg_clean")
                mlflow.log_metric("xgb_rmse", metrics_x.get("rmse", None))
                if metrics_x.get("rmse", float("inf")) < best_rmse:
                    best_rmse = metrics_x.get("rmse")
                    best_model = xgb
                    best_name = "xgboost"
                joblib.dump(xgb, OUT_DIR / "xgboost_regressor.pkl")
                mlflow.sklearn.log_model(xgb, "xgboost_regressor")
                print("[INFO] XGBoostRegressor saved & logged.")
            else:
                print("[INFO] XGBoost not available; skipped.")
        except Exception as e:
            print("[WARN] XGBoostRegressor failed:", e)

        # Save best model artifact
        try:
            if best_model is not None and best_name is not None:
                joblib.dump(best_model, OUT_DIR / f"best_regression_model_{best_name}.pkl")
                mlflow.log_param("best_model", best_name)
                mlflow.sklearn.log_model(best_model, f"best_{best_name}_regression_model")
                print(f"[INFO] Best regression model ({best_name}) saved to {OUT_DIR}")
        except Exception as e:
            print("[WARN] Saving best regression model failed:", e)

                # --- write used feature list for production (optional)
        try:
            from src.models.save_used_features import main as _save_used_features
            print("[INFO] Writing used feature list to models/used_feature_list.txt ...")
            _save_used_features()
        except Exception as _e:
            print("[WARN] Could not auto-save used feature list:", _e)


    print("[SUCCESS] Regression training (clean) finished.")

if __name__ == "__main__":
    run()
