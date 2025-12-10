# src/models/save_used_features.py

"""
This script extracts the final cleaned feature list used for inference.
It loads the processed data, applies the same cleaning (leakage removal),
and writes a stable feature list for ProductionModel.

Creates:
    models/used_feature_list.txt
"""

import pandas as pd
import os
from pathlib import Path

PROCESSED_PATH = "data/processed/processed_data.csv"
AUTO_DROP_PATH = "models/auto_dropped_features.txt"
OUTPUT_PATH = "src/models/used_feature_list.txt"

def load_auto_drop_list():
    """Reads auto_dropped_features.txt safely and strips BOM if present."""
    if not Path(AUTO_DROP_PATH).exists():
        return []

    drops = []
    with open(AUTO_DROP_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            col = line.strip().replace("\ufeff", "")
            if col:
                drops.append(col)
    return drops

def main():
    print("[INFO] Loading processed data...")

    if not Path(PROCESSED_PATH).exists():
        print(f"[ERROR] Processed file not found at {PROCESSED_PATH}")
        return

    df = pd.read_csv(PROCESSED_PATH)
    print(f"[INFO] Loaded shape: {df.shape}")

    # Read auto-drop list (columns detected as leakage)
    auto_drop = load_auto_drop_list()
    print("[INFO] Auto-drop list:", auto_drop)

    # Columns that must NEVER go into model
    known_leak_cols = [
        "Good_Investment",
        "Future_Price_5Y",
        "multi_factor_score",
        "pps_le_median_city",
        "price_le_median_city",
        "Price_Rs",
        "Price_in_Lakhs",       # price is user-input, but not a feature for classification
        "Price_per_SqFt"
    ]

    # Clean BOM-prefixed variants
    known_leak_cols = [c.replace("\ufeff", "") for c in known_leak_cols]

    # Combine unique set
    to_drop = set(auto_drop + known_leak_cols)

    print("[INFO] Removing leakage columns:", to_drop)
    df_clean = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    # Remove non-numeric original columns (since OHE produced numerics)
    non_numeric = [
        "Locality", "Public_Transport_Accessibility", "Parking_Space",
        "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status"
    ]
    df_clean = df_clean.drop(columns=[c for c in non_numeric if c in df_clean.columns], errors="ignore")

    # Final list of usable numeric features
    final_features = list(df_clean.select_dtypes(include=['int64', 'float64']).columns)

    print(f"[INFO] Final usable features: {len(final_features)}")

    # Save to file
    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for col in final_features:
            f.write(col + "\n")

    print(f"[SUCCESS] Saved feature list to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
