# src/models/inspect_feature_importance.py
"""
Correct inspector: loads raw dataset, applies SAME cleaning as training,
then computes feature importance and permutation importance correctly.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
from src.models.train_classification import load_and_clean, TARGET

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "classification_clean"

def run():
    print("[INFO] Loading data & applying SAME cleaning as model...")
    X, y = load_and_clean(DATA_PATH, TARGET)
    print(f"[INFO] Cleaned X.shape={X.shape}, y.shape={y.shape}")

    # Load best model
    model_files = list(MODEL_PATH.glob("best_classification_model_*.pkl"))
    if not model_files:
        print("[ERROR] No best model found in:", MODEL_PATH)
        return

    model_file = model_files[0]
    print("[INFO] Loading model:", model_file)
    model = joblib.load(model_file)

    feature_names = X.columns
    X_arr = X.values

    # ---------- Permutation Importance ----------
    print("[INFO] Calculating permutation importance...")
    perm = permutation_importance(model, X_arr, y, n_repeats=5, random_state=42, n_jobs=-1)

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std
    }).sort_values("perm_mean", ascending=False)

    out_dir = PROJECT_ROOT / "models" / "inspection"
    out_dir.mkdir(exist_ok=True)

    perm_path = out_dir / "permutation_importance_top30.csv"
    perm_df.head(30).to_csv(perm_path, index=False)

    print("\nTop 10 Permutation Importances:")
    print(perm_df.head(10))

    print(f"\n[SAVED] {perm_path}")

if __name__ == "__main__":
    run()
