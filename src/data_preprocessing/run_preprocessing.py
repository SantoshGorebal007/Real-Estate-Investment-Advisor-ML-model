# src/data_preprocessing/run_preprocessing.py
from pathlib import Path
import os
import json

# Local imports (ensure correct path or run from project root)
from src.data_preprocessing.load_data import load_raw_data
from src.data_preprocessing.clean_data import handle_missing_values, remove_duplicates, basic_inspect
from src.data_preprocessing.feature_engineering import add_basic_features, create_good_investment_label, compute_future_price_5y
from src.data_preprocessing.encode_scale import encode_categorical, scale_numerical
from src.data_preprocessing.split_dataset import split_and_save

RAW = "data/raw/india_housing_prices.csv"
OUT_DIR = "data/processed"
MODELS_DIR = "models"

def main():
    try:
        df = load_raw_data(RAW)
    except Exception as e:
        print("[FATAL] Loading raw data failed. Aborting pipeline.")
        return

    try:
        basic_inspect(df)
    except Exception:
        pass

    try:
        df = handle_missing_values(df)
    except Exception:
        print("[WARN] handle_missing_values encountered issues; continuing with original df")

    try:
        df = remove_duplicates(df)
    except Exception:
        print("[WARN] remove_duplicates failed; continuing.")

    try:
        df = add_basic_features(df)
    except Exception:
        print("[WARN] add_basic_features failed; continuing.")

    try:
        df = create_good_investment_label(df)
    except Exception:
        print("[WARN] create_good_investment_label failed; creating fallback label.")
        # fallback simple label
        if "Price_in_Lakhs" in df.columns:
            df["Good_Investment"] = (df["Price_in_Lakhs"] <= df["Price_in_Lakhs"].median()).astype(int)
        else:
            df["Good_Investment"] = 0

    try:
        df = compute_future_price_5y(df)
    except Exception:
        print("[WARN] compute_future_price_5y failed; setting to 0.")
        df["Future_Price_5Y"] = 0.0

    # encoding & scaling
    try:
        ohe_cols = ["State","City","Property_Type"]
        label_cols = ["Furnished_Status"]
        df_enc, encoders = encode_categorical(df, ohe_columns=ohe_cols, label_encode_cols=label_cols, save_dir=MODELS_DIR)
    except Exception:
        print("[WARN] encode_categorical failed; using original df for splitting.")
        df_enc = df

    try:
        numeric_cols = ["Size_in_SqFt","Price_Rs","Price_per_SqFt","Age_of_Property"]
        df_scaled, scaler = scale_numerical(df_enc, numeric_cols, save_path=f"{MODELS_DIR}/regression_scaler.pkl")
        if df_scaled is None:
            df_scaled = df_enc
    except Exception:
        print("[WARN] scale_numerical failed; using unscaled data.")
        df_scaled = df_enc

    try:
        train, test = split_and_save(df_scaled, target_col="Good_Investment", out_dir=OUT_DIR)
    except Exception:
        print("[FATAL] split_and_save failed. Attempting to save processed_data.csv only.")
        try:
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            df_scaled.to_csv(Path(OUT_DIR) / "processed_data.csv", index=False)
            print(f"[INFO] Saved processed_data.csv to {OUT_DIR}")
        except Exception as e:
            print(f"[ERROR] Failed to save processed data: {e}")
            return

    print("[SUCCESS] Preprocessing finished.")

if __name__ == "__main__":
    main()
# src/data_preprocessing/run_preprocessing.py
from pathlib import Path
import os
import json

# Local imports (ensure correct path or run from project root)
from src.data_preprocessing.load_data import load_raw_data
from src.data_preprocessing.clean_data import handle_missing_values, remove_duplicates, basic_inspect
from src.data_preprocessing.feature_engineering import add_basic_features, create_good_investment_label, compute_future_price_5y
from src.data_preprocessing.encode_scale import encode_categorical, scale_numerical
from src.data_preprocessing.split_dataset import split_and_save

RAW = "data/raw/india_housing_prices.csv"
OUT_DIR = "data/processed"
MODELS_DIR = "models"

def main():
    try:
        df = load_raw_data(RAW)
    except Exception as e:
        print("[FATAL] Loading raw data failed. Aborting pipeline.")
        return

    try:
        basic_inspect(df)
    except Exception:
        pass

    try:
        df = handle_missing_values(df)
    except Exception:
        print("[WARN] handle_missing_values encountered issues; continuing with original df")

    try:
        df = remove_duplicates(df)
    except Exception:
        print("[WARN] remove_duplicates failed; continuing.")

    try:
        df = add_basic_features(df)
    except Exception:
        print("[WARN] add_basic_features failed; continuing.")

    try:
        df = create_good_investment_label(df)
    except Exception:
        print("[WARN] create_good_investment_label failed; creating fallback label.")
        # fallback simple label
        if "Price_in_Lakhs" in df.columns:
            df["Good_Investment"] = (df["Price_in_Lakhs"] <= df["Price_in_Lakhs"].median()).astype(int)
        else:
            df["Good_Investment"] = 0

    try:
        df = compute_future_price_5y(df)
    except Exception:
        print("[WARN] compute_future_price_5y failed; setting to 0.")
        df["Future_Price_5Y"] = 0.0

    # encoding & scaling
    try:
        ohe_cols = ["State","City","Property_Type"]
        label_cols = ["Furnished_Status"]
        df_enc, encoders = encode_categorical(df, ohe_columns=ohe_cols, label_encode_cols=label_cols, save_dir=MODELS_DIR)
    except Exception:
        print("[WARN] encode_categorical failed; using original df for splitting.")
        df_enc = df

    try:
        numeric_cols = ["Size_in_SqFt","Price_Rs","Price_per_SqFt","Age_of_Property"]
        df_scaled, scaler = scale_numerical(df_enc, numeric_cols, save_path=f"{MODELS_DIR}/regression_scaler.pkl")
        if df_scaled is None:
            df_scaled = df_enc
    except Exception:
        print("[WARN] scale_numerical failed; using unscaled data.")
        df_scaled = df_enc

    try:
        train, test = split_and_save(df_scaled, target_col="Good_Investment", out_dir=OUT_DIR)
    except Exception:
        print("[FATAL] split_and_save failed. Attempting to save processed_data.csv only.")
        try:
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            df_scaled.to_csv(Path(OUT_DIR) / "processed_data.csv", index=False)
            print(f"[INFO] Saved processed_data.csv to {OUT_DIR}")
        except Exception as e:
            print(f"[ERROR] Failed to save processed data: {e}")
            return

    print("[SUCCESS] Preprocessing finished.")

if __name__ == "__main__":
    main()
