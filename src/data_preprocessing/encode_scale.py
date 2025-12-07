# src/data_preprocessing/encode_scale.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def encode_categorical(df: pd.DataFrame, ohe_columns: list = None, label_encode_cols: list = None, save_dir: str = "models"):
    """
    Robust categorical encoding:
    - label_encode_cols: uses pd.factorize and saves mapping as joblib
    - ohe_columns: tries sklearn.OneHotEncoder with modern param names, falls back to pandas.get_dummies
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    encoders = {}

    df = df.copy()

    # Label encode via factorize
    if label_encode_cols:
        for col in label_encode_cols:
            try:
                vals, uniques = pd.factorize(df[col].astype(str))
                df[col] = vals
                mapping = {k: i for i, k in enumerate(uniques)}
                encoders[col] = mapping
                joblib.dump(mapping, Path(save_dir) / f"{col}_label_encoder.pkl")
            except Exception as e:
                print(f"[WARN] Label encode failed for {col}: {e}")

    # One-hot via sklearn if available and compatible, else pandas.get_dummies
    if ohe_columns:
        try:
            # try sklearn OneHotEncoder with modern api
            from sklearn.preprocessing import OneHotEncoder
            try:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                # older scikit-learn expects 'sparse'
                try:
                    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
                except TypeError:
                    raise

            # fit-transform
            ohe_fit = ohe.fit(df[ohe_columns].astype(str))
            ohe_cols = ohe_fit.get_feature_names_out(ohe_columns)
            ohe_arr = ohe_fit.transform(df[ohe_columns].astype(str))
            # ensure we can handle both dense array and sparse matrix
            if hasattr(ohe_arr, "toarray"):
                ohe_arr = ohe_arr.toarray()
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
            df = pd.concat([df.drop(columns=ohe_columns, errors='ignore'), ohe_df], axis=1)
            encoders["onehot"] = ohe_fit
            joblib.dump(ohe_fit, Path(save_dir) / "onehot_encoder.pkl")
        except Exception as e:
            print(f"[WARN] OneHotEncoder sklearn failed or incompatible: {e}")
            # fallback to pandas
            try:
                df = pd.get_dummies(df, columns=ohe_columns, dummy_na=False)
                # save the list of columns that were created so we can align new data later
                ohe_cols = [c for c in df.columns if any(orig + "_" in c for orig in ohe_columns)]
                encoders["onehot_columns"] = ohe_cols
                joblib.dump(ohe_cols, Path(save_dir) / "onehot_columns_list.pkl")
                print("[INFO] Fallback: used pandas.get_dummies for OHE.")
            except Exception as e2:
                print(f"[ERROR] pandas.get_dummies also failed: {e2}")

    return df, encoders

def scale_numerical(df: pd.DataFrame, numeric_cols: list, save_path: str = "models/regression_scaler.pkl"):
    """
    Lightweight scaler: zero-mean, unit-variance computed and saved as dict.
    """
    df = df.copy()
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        print("[WARN] No numeric columns found to scale.")
        return df, None

    stats = {}
    for col in numeric_cols:
        try:
            col_vals = pd.to_numeric(df[col], errors="coerce")
            mean = float(col_vals.mean(skipna=True))
            std = float(col_vals.std(ddof=0)) if float(col_vals.std(ddof=0)) > 0 else 1.0
            stats[col] = {"mean": mean, "std": std}
            df[col] = (col_vals - mean) / std
        except Exception as e:
            print(f"[WARN] Scaling failed for {col}: {e}")

    try:
        joblib.dump(stats, save_path)
    except Exception as e:
        print(f"[WARN] Failed to save scaler stats: {e}")

    return df, stats
