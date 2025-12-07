# src/data_preprocessing/clean_data.py
import pandas as pd
import numpy as np

def basic_inspect(df: pd.DataFrame) -> None:
    try:
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Missing by column:\n", df.isna().sum())
    except Exception as e:
        print(f"[WARN] basic_inspect failed: {e}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        # numeric fill: median
        num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        for c in num_cols:
            if df[c].isna().sum() > 0:
                df[c] = df[c].fillna(df[c].median())

        # categorical fill: mode (if mode exists)
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in cat_cols:
            try:
                if df[c].isna().sum() > 0:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
            except Exception:
                # fallback: fill with placeholder
                df[c] = df[c].fillna("Unknown")

        # Price_per_SqFt fallback compute
        if "Price_per_SqFt" in df.columns:
            try:
                mask = df["Price_per_SqFt"].isna() & df["Price_in_Lakhs"].notna() & df["Size_in_SqFt"].notna()
                df.loc[mask, "Price_per_SqFt"] = (df.loc[mask, "Price_in_Lakhs"] * 100000) / df.loc[mask, "Size_in_SqFt"]
            except Exception as e:
                print(f"[WARN] Price_per_SqFt compute failed: {e}")

        return df
    except Exception as e:
        print(f"[ERROR] handle_missing_values failed: {e}")
        raise

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        after = len(df)
        print(f"Removed duplicates: {before-after}")
        return df
    except Exception as e:
        print(f"[ERROR] remove_duplicates failed: {e}")
        raise
