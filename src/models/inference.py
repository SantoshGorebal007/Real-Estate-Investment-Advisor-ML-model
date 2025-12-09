# src/models/inference.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import sys

class ProductionModel:
    def __init__(self,
                 classification_path: str = None,
                 regression_path: str = None,
                 feature_list_path: str = None):
        base = Path(__file__).resolve().parents[2]

        # default paths
        self.classification_path = Path(classification_path) if classification_path else base / "models" / "classification_clean" / "best_classification_model_random_forest.pkl"
        self.regression_path = Path(regression_path) if regression_path else base / "models" / "regression_clean" / "best_regression_model_random_forest.pkl"
        self.feature_list_path = Path(feature_list_path) if feature_list_path else base / "models" / "used_feature_list.txt"

        # load feature list (fallback)
        if not self.feature_list_path.exists():
            raise FileNotFoundError(f"Feature list not found: {self.feature_list_path}. Run save_used_features.py first.")
        with open(self.feature_list_path, "r", encoding="utf-8") as f:
            self.used_features = [ln.strip() for ln in f.readlines() if ln.strip()]

        # alias for backward compatibility
        self.feature_list = self.used_features

        # load classifier/regressor
        if not self.classification_path.exists():
            raise FileNotFoundError(f"Classifier not found: {self.classification_path}")
        self.classifier = joblib.load(self.classification_path)

        if not self.regression_path.exists():
            raise FileNotFoundError(f"Regressor not found: {self.regression_path}")
        self.regressor = joblib.load(self.regression_path)

        # If model exposes feature_names_in_, use it as authoritative list
        self.model_feature_names = None
        try:
            if hasattr(self.classifier, "feature_names_in_"):
                self.model_feature_names = list(self.classifier.feature_names_in_)
        except Exception:
            self.model_feature_names = None

        # final authoritative feature list (model first, fallback to saved list)
        if self.model_feature_names:
            self.authoritative_features = self.model_feature_names
        else:
            self.authoritative_features = self.used_features

    def _compute_price_per_sqft_if_missing(self, df: pd.DataFrame):
        # Compute Price_per_SqFt if possible otherwise add zero column
        if "Price_per_SqFt" not in df.columns:
            if ("Price_in_Lakhs" in df.columns) and ("Size_in_SqFt" in df.columns):
                with np.errstate(divide='ignore', invalid='ignore'):
                    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000.0) / df["Size_in_SqFt"]
                    df["Price_per_SqFt"] = df["Price_per_SqFt"].replace([np.inf, -np.inf], 0).fillna(0.0)
            else:
                df["Price_per_SqFt"] = 0.0
        return df

    def _coerce_to_df(self, X):
        """Accept dict / DataFrame / list / ndarray and return DataFrame"""
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, (list, tuple, np.ndarray)):
            df = pd.DataFrame(X)
        else:
            raise ValueError("Input must be dict, DataFrame, list or numpy array.")
        return df

    def _sanitize_columns(self, df: pd.DataFrame):
        # 1) drop index-like / id-like columns (case-insensitive)
        banned_lower = {"id", "index"}
        drop_cols = [c for c in df.columns if c.strip().lower() in banned_lower]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        # 2) compute derived features expected by model
        df = self._compute_price_per_sqft_if_missing(df)

        # 3) Drop any columns that the model did NOT see at training time (unexpected)
        # Use authoritative_features (from model.feature_names_in_ if available)
        unexpected = [c for c in df.columns if c not in self.authoritative_features]
        if unexpected:
            df = df.drop(columns=unexpected, errors="ignore")

        # 4) Add any missing features (the ones model expects) with zeros
        for c in self.authoritative_features:
            if c not in df.columns:
                df[c] = 0.0

        # 5) Reorder to exactly match authoritative order
        df = df[self.authoritative_features]

        # 6) Ensure numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        return df

    def _prepare(self, X):
        df = self._coerce_to_df(X)
        df = self._sanitize_columns(df)
        return df

    def predict_investment(self, data):
        """
        Returns (label:int, prob:float or None)
        """
        X = self._prepare(data)
        # final check: ensure columns match model.feature_names_in_ if present
        if hasattr(self.classifier, "feature_names_in_"):
            # scikit-learn will still validate precisely; we already aligned but double-check lengths
            model_names = list(self.classifier.feature_names_in_)
            if list(X.columns) != model_names:
                # attempt to reorder to model_names (shouldn't happen if authoritative_features from model)
                missing = [c for c in model_names if c not in X.columns]
                extra = [c for c in X.columns if c not in model_names]
                raise ValueError(f"Post-processed feature mismatch. Missing: {missing}. Extra: {extra}")
        label = int(self.classifier.predict(X)[0])
        proba = None
        if hasattr(self.classifier, "predict_proba"):
            try:
                probs = self.classifier.predict_proba(X)
                proba = float(probs[0,1]) if probs.shape[1] > 1 else float(probs[0,0])
            except Exception:
                proba = None
        return label, proba

    def predict_future_price(self, data):
        X = self._prepare(data)
        # same check as above
        pred = float(self.regressor.predict(X)[0])
        return pred

    def predict_all(self, data):
        label, prob = self.predict_investment(data)
        future = self.predict_future_price(data)
        return {
            "good_investment": int(label),
            "confidence": float(prob) if prob is not None else None,
            "future_price_5y": float(future)
        }

    def predict(self, data):
        return self.predict_all(data)


# quick test block when run directly
if __name__ == "__main__":
    pm = ProductionModel()
    sample = {"BHK":3, "Size_in_SqFt":1200, "Price_in_Lakhs":80}
    print("Authoritative features count:", len(pm.authoritative_features))
    print(pm.predict_all(sample))
