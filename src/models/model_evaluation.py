# src/models/model_evaluation.py
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns

PLOTS_DIR = Path("src/eda/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(fig, name):
    out = PLOTS_DIR / name
    try:
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f"[SAVED] {out}")
    except Exception as e:
        print(f"[WARN] could not save plot {name}: {e}")

# --- Classification metrics & plots ---
def evaluate_classification(y_true, y_pred, y_proba=None, prefix="clf"):
    try:
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None

        print("Classification metrics:")
        for k,v in metrics.items():
            print(f"  {k}: {v}")

        # confusion matrix
        fig, ax = plt.subplots(figsize=(5,4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{prefix} Confusion Matrix")
        save_plot(fig, f"{prefix}_confusion_matrix.png")
        plt.close(fig)

        # ROC curve
        if y_proba is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                fig2, ax2 = plt.subplots(figsize=(5,4))
                ax2.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}" if metrics['roc_auc'] else "ROC")
                ax2.plot([0,1],[0,1], linestyle='--', color='gray')
                ax2.set_xlabel("FPR")
                ax2.set_ylabel("TPR")
                ax2.set_title(f"{prefix} ROC Curve")
                ax2.legend()
                save_plot(fig2, f"{prefix}_roc_curve.png")
                plt.close(fig2)
            except Exception as e:
                print("[WARN] ROC plot failed:", e)

        return metrics
    except Exception as e:
        print("[ERROR] evaluate_classification failed:", e)
        return {}

# --- Regression metrics & plots ---
def evaluate_regression(y_true, y_pred, prefix="reg"):
    try:
        metrics = {}
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))

        print("Regression metrics:")
        for k,v in metrics.items():
            print(f"  {k}: {v}")

        # residuals plot
        fig, ax = plt.subplots(figsize=(6,4))
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title(f"{prefix} Residuals Distribution")
        save_plot(fig, f"{prefix}_residuals_hist.png")
        plt.close(fig)

        # y_true vs y_pred scatter
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.scatter(y_true, y_pred, alpha=0.2)
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_title(f"{prefix} Actual vs Predicted")
        save_plot(fig2, f"{prefix}_actual_vs_predicted.png")
        plt.close(fig2)

        return metrics
    except Exception as e:
        print("[ERROR] evaluate_regression failed:", e)
        return {}
