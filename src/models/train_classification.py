# src/models/train_classification_clean.py
"""
Leak-safe classification training script.
Drops columns that likely leak the target before training.
Also reads models/auto_dropped_features.txt to remove explicit columns.
Trains RandomForest (+ XGBoost if available), logs to MLflow,
saves best model to models/classification_clean/.
"""
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.models.model_evaluation import evaluate_classification

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Paths / constants
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
OUT_DIR = PROJECT_ROOT / "models" / "classification_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Good_Investment"

# Patterns to detect likely leakage / target-derived features (lowercase checked)
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
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_xgb(X_train, y_train):
    if not HAVE_XGB:
        return None
    xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    return xgb

def run():
    try:
        print("[INFO] Loading and cleaning data...")
        X, y = load_and_clean()
    except Exception as e:
        print("[FATAL] Failed to load/clean data:", e)
        return

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    best_model, best_name, best_score = None, None, -1.0

    mlflow.set_experiment("RealEstate_Classification_Clean")
    with mlflow.start_run(run_name="baseline_clean"):
        # Random Forest
        try:
            print("[INFO] Training RandomForestClassifier...")
            rf = train_rf(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test) if hasattr(rf, "predict_proba") else None
            metrics_rf = evaluate_classification(y_test, y_pred, y_proba, prefix="rf_clean")
            mlflow.log_metric("rf_f1", metrics_rf.get("f1", 0))
            if metrics_rf.get("f1", 0) > best_score:
                best_score = metrics_rf.get("f1", 0)
                best_model = rf
                best_name = "random_forest"
            joblib.dump(rf, OUT_DIR / "random_forest_classifier.pkl")
            mlflow.sklearn.log_model(rf, "random_forest_classifier")
            print("[INFO] RandomForest saved & logged.")
        except Exception as e:
            print("[WARN] RF training failed:", e)

        # XGBoost
        try:
            if HAVE_XGB:
                print("[INFO] Training XGBoostClassifier...")
                xgb = train_xgb(X_train, y_train)
                y_pred_x = xgb.predict(X_test)
                y_proba_x = xgb.predict_proba(X_test) if hasattr(xgb, "predict_proba") else None
                metrics_x = evaluate_classification(y_test, y_pred_x, y_proba_x, prefix="xgb_clf_clean")
                mlflow.log_metric("xgb_f1", metrics_x.get("f1", 0))
                if metrics_x.get("f1", 0) > best_score:
                    best_score = metrics_x.get("f1", 0)
                    best_model = xgb
                    best_name = "xgboost"
                joblib.dump(xgb, OUT_DIR / "xgboost_classifier.pkl")
                mlflow.sklearn.log_model(xgb, "xgboost_classifier")
                print("[INFO] XGBoost saved & logged.")
            else:
                print("[INFO] XGBoost not available; skipped.")
        except Exception as e:
            print("[WARN] XGBoost training failed:", e)

        # Save best model artifact
        try:
            if best_model is not None and best_name is not None:
                joblib.dump(best_model, OUT_DIR / f"best_classification_model_{best_name}.pkl")
                mlflow.log_param("best_model", best_name)
                mlflow.sklearn.log_model(best_model, f"best_{best_name}_model")
                print(f"[INFO] Best model ({best_name}) saved to {OUT_DIR}")
        except Exception as e:
            print("[WARN] Saving best model failed:", e)
        
                # --- write used feature list for production (optional)
        try:
            from src.models.save_used_features import main as _save_used_features
            print("[INFO] Writing used feature list to models/used_feature_list.txt ...")
            _save_used_features()
        except Exception as _e:
            print("[WARN] Could not auto-save used feature list:", _e)


    print("[SUCCESS] Classification training (clean) finished.")

if __name__ == "__main__":
    run()
