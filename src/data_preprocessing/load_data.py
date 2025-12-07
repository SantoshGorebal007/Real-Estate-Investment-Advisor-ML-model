# src/data_preprocessing/load_data.py
import pandas as pd
from pathlib import Path

def load_raw_data(path: str) -> pd.DataFrame:
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found at: {path}")
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        raise
    except pd.errors.EmptyDataError:
        print(f"[ERROR] File is empty: {path}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error loading data: {e}")
        raise

if __name__ == "__main__":
    df = load_raw_data("data/raw/india_housing_prices.csv")
    print("Loaded rows:", len(df))
    print(df.head(3))
