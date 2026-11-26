"""
Scratch script untuk mendapatkan metrik aktual XGBoost
yang akan dimasukkan ke notebook dan README.
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

# 1. Path relatif ke folder data/processed
PROCESSED_DIR = Path("../data/processed")
if not PROCESSED_DIR.exists():
    PROCESSED_DIR = Path("data/processed")

print(f"Data dir: {PROCESSED_DIR.resolve()}")

# 2. Load Dataset Train & Validation
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()

X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv").values.ravel()

# 3. Load Class Weights JSON
with open(PROCESSED_DIR / "class_weights.json", "r") as f:
    class_weights_raw = json.load(f)

# Mapping Kunci JSON ke Integer (0, 1, 2, 3)
label_mapping = {"Low": 0, "Moderate": 1, "High": 2, "Extreme": 3}
class_weights = {}
for k, v in class_weights_raw.items():
    if k in label_mapping:
        class_weights[label_mapping[k]] = float(v)
    else:
        class_weights[int(k)] = float(v)

print(f"Data Training   : {X_train.shape[0]} sampel, {X_train.shape[1]} fitur")
print(f"Data Validation : {X_val.shape[0]} sampel, {X_val.shape[1]} fitur")
print(f"Class Weights   : {class_weights}")
print(f"NaN Training    : {X_train.isnull().sum().sum()}")
print(f"NaN Validation  : {X_val.isnull().sum().sum()}")

# 4. Bangun array sample_weight_train
sample_weight_train = np.array([class_weights[label] for label in y_train])
print(f"\nSample weight stats:")
print(f"  min={sample_weight_train.min():.4f}, max={sample_weight_train.max():.4f}")
print(f"  unique weights: {np.unique(sample_weight_train)}")

# 5. Latih XGBClassifier
xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# Waktu training
start_train = time.time()
xgb_model.fit(X_train, y_train, sample_weight=sample_weight_train)
end_train = time.time()
train_duration = end_train - start_train

# 6. Inferensi dan waktu inferensi
start_infer = time.time()
y_pred_val = xgb_model.predict(X_val)
end_infer = time.time()
infer_duration = end_infer - start_infer
latency_per_sample = (infer_duration / len(X_val)) * 1000

# 7. Hitung metrik Train
y_pred_train = xgb_model.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
macro_f1_train = f1_score(y_train, y_pred_train, average="macro")

# 8. Hitung metrik Validation
acc_val = accuracy_score(y_val, y_pred_val)
macro_f1_val = f1_score(y_val, y_pred_val, average="macro")
weighted_f1_val = f1_score(y_val, y_pred_val, average="weighted")

# Per-kelas
report_dict = {}
from sklearn.metrics import precision_recall_fscore_support
precisions, recalls, f1s, supports = precision_recall_fscore_support(y_val, y_pred_val, labels=[0,1,2,3])
recall_high = recalls[2]
recall_extreme = recalls[3]
recall_high_extreme_avg = (recalls[2] + recalls[3]) / 2

# 9. Overfitting check
gap = abs(macro_f1_train - macro_f1_val)
if gap >= 0.20:
    status_overfit = "Mengalami Overfitting Parah"
elif gap >= 0.10:
    status_overfit = "Mengalami Overfitting Moderat"
else:
    status_overfit = "Model Stabil (Good Fit)"

# 10. Tampilkan hasil
print("=" * 60)
print("ANALISIS LENGKAP MODEL ADVANCED — XGBoost")
print("=" * 60)
print(f"Validation Accuracy      : {acc_val:.4f} ({acc_val*100:.2f}%)")
print(f"Validation Macro F1      : {macro_f1_val:.4f}")
print(f"Validation Weighted F1   : {weighted_f1_val:.4f}")
print("-" * 60)
print(f"Training Macro F1        : {macro_f1_train:.4f} (Accuracy: {acc_train*100:.2f}%)")
print(f"Train-Val Gap (Overfit)  : {gap:.4f} ({status_overfit})")
print(f"Waktu Training           : {train_duration:.4f} detik")
print(f"Waktu Inferensi Total    : {infer_duration:.4f} detik ({latency_per_sample:.4f} ms/sample)")
print("-" * 60)
print(f"\nRecall kelas High (2)    : {recall_high:.4f}")
print(f"Recall kelas Extreme (3) : {recall_extreme:.4f}")
print(f"Avg Recall High+Extreme  : {recall_high_extreme_avg:.4f}")
print()

target_names = [
    "Class 0 (Low)",
    "Class 1 (Moderate)",
    "Class 2 (High)",
    "Class 3 (Extreme)",
]
print("Classification Report (Validation Set 2025):\n")
print(classification_report(y_val, y_pred_val, target_names=target_names))

# 11. Simpan Confusion Matrix
os.makedirs("../reports/figures", exist_ok=True)
if not os.path.exists("../reports/figures"):
    os.makedirs("reports/figures", exist_ok=True)
    fig_dir = Path("reports/figures")
else:
    fig_dir = Path("../reports/figures")

cm = confusion_matrix(y_val, y_pred_val)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Moderate", "High", "Extreme"],
    yticklabels=["Low", "Moderate", "High", "Extreme"],
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — XGBoost (Validation Set 2025)", fontsize=13)
plt.tight_layout()

cm_path = fig_dir / "cm_val_xgboost_initial.png"
plt.savefig(str(cm_path), dpi=100, bbox_inches="tight")
print(f"Confusion Matrix saved to: {cm_path}")
plt.close()

# 12. Feature Importance (Top 15)
importances = xgb_model.feature_importances_
feature_names = X_train.columns.tolist()
feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).head(15)

fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis", ax=ax2)
ax2.set_xlabel("Importance (Weight)", fontsize=12)
ax2.set_ylabel("Feature", fontsize=12)
ax2.set_title("Top 15 Feature Importance — XGBoost", fontsize=13)
plt.tight_layout()

fi_path = fig_dir / "fi_xgboost_initial.png"
plt.savefig(str(fi_path), dpi=100, bbox_inches="tight")
print(f"Feature Importance saved to: {fi_path}")
plt.close()

print("\n=== TOP 15 FEATURES ===")
print(feat_imp_df.to_string(index=False))
