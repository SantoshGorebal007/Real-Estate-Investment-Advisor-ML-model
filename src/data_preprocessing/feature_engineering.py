# src/data_preprocessing/feature_engineering.py
import pandas as pd
import numpy as np

CURRENT_YEAR = 2025  # change if needed

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        # Age if Year_Built available
        if "Year_Built" in df.columns:
            try:
                df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
            except Exception:
                df["Age_of_Property"] = np.nan
        elif "Age_of_Property" not in df.columns:
            df["Age_of_Property"] = np.nan

        # Price_per_SqFt recompute fallback
        if "Price_per_SqFt" not in df.columns or df["Price_per_SqFt"].isna().all():
            if "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
                    df["Price_per_SqFt"] = df["Price_per_SqFt"].replace([np.inf, -np.inf], np.nan)
            else:
                df["Price_per_SqFt"] = np.nan

        # Price absolute in rupees
        if "Price_in_Lakhs" in df.columns:
            df["Price_Rs"] = df["Price_in_Lakhs"] * 100000
        else:
            df["Price_Rs"] = np.nan

        # Normalize counts to density-like score: simple min-max for schools/hospitals
        for col in ["Nearby_Schools", "Nearby_Hospitals"]:
            if col in df.columns:
                try:
                    minv = df[col].min()
                    maxv = df[col].max()
                    if pd.notna(minv) and pd.notna(maxv) and maxv > minv:
                        df[col + "_score"] = (df[col] - minv) / (maxv - minv)
                    else:
                        df[col + "_score"] = 0.0
                except Exception:
                    df[col + "_score"] = 0.0

        # Price per sqft zscore
        if "Price_per_SqFt" in df.columns:
            try:
                mean = df["Price_per_SqFt"].mean()
                std = df["Price_per_SqFt"].std(ddof=0)
                df["pps_zscore"] = (df["Price_per_SqFt"] - mean) / (std + 1e-9)
            except Exception:
                df["pps_zscore"] = 0.0

        return df
    except Exception as e:
        print(f"[ERROR] add_basic_features failed: {e}")
        raise

def create_good_investment_label(df: pd.DataFrame,
                                 city_col: str = "City",
                                 price_col: str = "Price_in_Lakhs",
                                 pps_col: str = "Price_per_SqFt",
                                 multi_factor_threshold: float = 0.6) -> pd.DataFrame:
    try:
        df = df.copy()

        # city median price & pps
        if city_col in df.columns and price_col in df.columns:
            try:
                city_med_price = df.groupby(city_col)[price_col].transform("median")
                df["price_le_median_city"] = (df[price_col] <= city_med_price).astype(int)
            except Exception:
                df["price_le_median_city"] = 0
        else:
            df["price_le_median_city"] = 0

        if city_col in df.columns and pps_col in df.columns:
            try:
                city_med_pps = df.groupby(city_col)[pps_col].transform("median")
                df["pps_le_median_city"] = (df[pps_col] <= city_med_pps).astype(int)
            except Exception:
                df["pps_le_median_city"] = 0
        else:
            df["pps_le_median_city"] = 0

        # simple multi-factor: BHK>=3, Availability_Status == 'Available', RERA-like field if exists
        if "BHK" in df.columns:
            try:
                df["bhk_good"] = (df["BHK"] >= 3).astype(int)
            except Exception:
                df["bhk_good"] = 0
        else:
            df["bhk_good"] = 0

        if "Availability_Status" in df.columns:
            try:
                df["available_good"] = (df["Availability_Status"].str.lower() == "available").astype(int)
            except Exception:
                df["available_good"] = 0
        else:
            df["available_good"] = 0

        if "RERA" in df.columns:
            try:
                df["rera_good"] = df["RERA"].astype(int)
            except Exception:
                df["rera_good"] = 0
        else:
            df["rera_good"] = 0

        # combine: weighted sum (simple average here)
        cols = ["price_le_median_city", "pps_le_median_city", "bhk_good", "available_good", "rera_good"]
        present_cols = [c for c in cols if c in df.columns]
        if present_cols:
            df["multi_factor_score"] = df[present_cols].sum(axis=1) / len(present_cols)
        else:
            df["multi_factor_score"] = 0.0

        df["Good_Investment"] = (df["multi_factor_score"] >= multi_factor_threshold).astype(int)

        return df
    except Exception as e:
        print(f"[ERROR] create_good_investment_label failed: {e}")
        raise

def compute_future_price_5y(df: pd.DataFrame,
                            base_price_col: str = "Price_in_Lakhs",
                            growth_rate_default: float = 0.08,
                            growth_by_city: dict = None) -> pd.DataFrame:
    try:
        df = df.copy()
        if growth_by_city is None:
            growth_by_city = {}

        def get_rate(row):
            try:
                city = row.get("City")
                return growth_by_city.get(city, growth_rate_default)
            except Exception:
                return growth_rate_default

        if base_price_col not in df.columns:
            df[base_price_col] = 0.0

        rates = df.apply(get_rate, axis=1)
        with np.errstate(all='ignore'):
            df["Future_Price_5Y"] = df[base_price_col] * ((1 + rates) ** 5)

        df["Future_Price_5Y"] = df["Future_Price_5Y"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df
    except Exception as e:
        print(f"[ERROR] compute_future_price_5y failed: {e}")
        raise
