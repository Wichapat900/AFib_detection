"""
test_xgb.py
===========
Evaluates XGBoost separately to avoid macOS segfault.
Run this before test.py.
Saves results/xgb_result.json and results/xgb_probs.npy
"""

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, fbeta_score, confusion_matrix
)
from sklearn.impute import SimpleImputer

Path("results").mkdir(exist_ok=True)

# ============================================================
# LOAD
# ============================================================

hrv_test = np.load("data/mitb/hrv_X_test.npy")
hrv_test = np.nan_to_num(hrv_test)
y_test   = np.load("data/mitb/y_test.npy")

print(f"Test: {hrv_test.shape}  AFib={int((y_test==1).sum())}  Normal={int((y_test==0).sum())}")

# ============================================================
# EVALUATE
# ============================================================

bundle = joblib.load("models/xgb.pkl")
model  = bundle["model"]
thresh = bundle["threshold"]
model.n_jobs = 1

imp   = bundle.get("imputer", SimpleImputer(strategy="median"))
X_imp = imp.transform(hrv_test)
probs = model.predict_proba(X_imp)[:, 1]
preds = (probs >= thresh).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

result = {
    "model":       "XGBoost",
    "auroc":       float(roc_auc_score(y_test, probs)),
    "f1":          float(f1_score(y_test, preds, zero_division=0)),
    "f2":          float(fbeta_score(y_test, preds, beta=2, zero_division=0)),
    "recall":      float(recall_score(y_test, preds, zero_division=0)),
    "precision":   float(precision_score(y_test, preds, zero_division=0)),
    "sensitivity": float(tp / (tp + fn + 1e-8)),
    "specificity": float(tn / (tn + fp + 1e-8)),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "threshold":   float(thresh),
}

print(f"\nXGBoost Results:")
print(f"  AUROC={result['auroc']:.4f}  Recall={result['recall']:.4f}  "
      f"F2={result['f2']:.4f}  Spec={result['specificity']:.4f}")
print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  TN={result['tn']}")

# ============================================================
# SAVE
# ============================================================

np.save("results/xgb_probs.npy", probs)
with open("results/xgb_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved results/xgb_result.json")
print("Saved results/xgb_probs.npy")
print("\nNow run: python test.py")