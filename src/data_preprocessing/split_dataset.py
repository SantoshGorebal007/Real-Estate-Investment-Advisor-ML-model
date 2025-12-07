# src/data_preprocessing/split_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_and_save(df: pd.DataFrame, target_col: str, test_size: float = 0.2, out_dir: str = "data/processed"):
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(Path(out_dir) / "train.csv", index=False)
        test.to_csv(Path(out_dir) / "test.csv", index=False)
        df.to_csv(Path(out_dir) / "processed_data.csv", index=False)
        print(f"[INFO] Saved processed files to {out_dir}")
        return train, test
    except KeyError as e:
        print(f"[ERROR] split_and_save failed: {e}")
        raise
    except ValueError as e:
        print(f"[ERROR] split_and_save value error (maybe stratify issue): {e}")
        raise
    except Exception as e:
        print(f"[ERROR] split_and_save unexpected error: {e}")
        raise
# src/data_preprocessing/split_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_and_save(df: pd.DataFrame, target_col: str, test_size: float = 0.2, out_dir: str = "data/processed"):
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(Path(out_dir) / "train.csv", index=False)
        test.to_csv(Path(out_dir) / "test.csv", index=False)
        df.to_csv(Path(out_dir) / "processed_data.csv", index=False)
        print(f"[INFO] Saved processed files to {out_dir}")
        return train, test
    except KeyError as e:
        print(f"[ERROR] split_and_save failed: {e}")
        raise
    except ValueError as e:
        print(f"[ERROR] split_and_save value error (maybe stratify issue): {e}")
        raise
    except Exception as e:
        print(f"[ERROR] split_and_save unexpected error: {e}")
        raise
