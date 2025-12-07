# src/data_preprocessing/encode_scale.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pathlib import Path
import joblib

def encode_categorical(df: pd.DataFrame, ohe_columns: list = None, label_encode_cols: list = None, save_dir: str = "models") -> (pd.DataFrame, dict):
    encoders = {}
    try:
        df = df.copy()
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if label_encode_cols:
            for col in label_encode_cols:
                try:
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le
                    joblib.dump(le, Path(save_dir) / f"{col}_label_encoder.pkl")
                except Exception as e:
                    print(f"[WARN] Label encoding failed for {col}: {e}")

        if ohe_columns:
            try:
                ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
                ohe_fit = ohe.fit(df[ohe_columns].astype(str))
                ohe_cols = ohe_fit.get_feature_names_out(ohe_columns)
                ohe_arr = ohe_fit.transform(df[ohe_columns].astype(str))
                ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
                df = pd.concat([df.drop(columns=ohe_columns, errors='ignore'), ohe_df], axis=1)
                encoders["onehot"] = ohe_fit
                joblib.dump(ohe_fit, Path(save_dir) / "onehot_encoder.pkl")
            except Exception as e:
                print(f"[WARN] OneHot encoding failed: {e}")

        return df, encoders
    except Exception as e:
        print(f"[ERROR] encode_categorical failed: {e}")
        raise

def scale_numerical(df: pd.DataFrame, numeric_cols: list, save_path: str = "models/regression_scaler.pkl"):
    try:
        df = df.copy()
        # ensure numeric cols exist
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        if not numeric_cols:
            print("[WARN] No numeric columns found to scale.")
            return df, None

        scaler = StandardScaler()
        try:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(float))
            joblib.dump(scaler, save_path)
            return df, scaler
        except Exception as e:
            print(f"[WARN] Scaling failed: {e}")
            return df, None
    except Exception as e:
        print(f"[ERROR] scale_numerical failed: {e}")
        raise
