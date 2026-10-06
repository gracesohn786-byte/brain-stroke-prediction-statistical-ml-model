
"""
Step 6: Test-set evaluation of the saved default decision tree model.

Restrictions followed:
- The saved model is loaded and evaluated as-is.
- No retraining, tuning, pruning, class weighting, or resampling is performed.
- The ROC-based threshold is selected from the test ROC curve and is therefore exploratory.
"""

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path("/mnt/data")
TRAIN_PATH = BASE_DIR / "train_dataset.csv"
TEST_PATH = BASE_DIR / "test_dataset.csv"
MODEL_PATH = BASE_DIR / "default_decision_tree_model.pkl"
TARGET = "stroke"

# 1-3. Load datasets and saved model.
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)

# 4-5. Separate features and target; remove ID-like columns consistently if present.
id_like_columns = [
    c for c in train_df.columns
    if c.lower() in ["id", "patient_id", "patientid", "record_id", "recordid"]
]

X_train = train_df.drop(columns=[TARGET] + id_like_columns)
y_train = train_df[TARGET]

test_id_like_columns = [c for c in id_like_columns if c in test_df.columns]
X_test = test_df.drop(columns=[TARGET] + test_id_like_columns)
y_test = test_df[TARGET]

# 6. Generate predicted probabilities for stroke = 1.
y_prob = model.predict_proba(X_test)[:, 1]

def metric_row(y_true, y_prob, threshold):
    """Calculate threshold-dependent and threshold-independent metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc_average_precision": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# 7-8. Default threshold evaluation.
metrics_050 = metric_row(y_test, y_prob, 0.50)

# 9-12. ROC curve and Youden's J threshold.
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
specificity_values = 1 - fpr
youden_j = tpr + specificity_values - 1
best_idx = int(np.argmax(youden_j))
best_threshold = float(roc_thresholds[best_idx])

# Save ROC curve.
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Default decision tree, AUC = {metrics_050['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.scatter(
    fpr[best_idx],
    tpr[best_idx],
    label=f"Exploratory Youden threshold = {best_threshold:.3f}",
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("Default Decision Tree Test ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(BASE_DIR / "default_decision_tree_test_roc_curve.png", dpi=200)
plt.close()

# 13-16. Best-threshold and threshold comparison metrics.
metrics_best = metric_row(y_test, y_prob, best_threshold)

thresholds_to_evaluate = [0.10, 0.20, 0.30, 0.40, 0.50, best_threshold]
unique_thresholds = []
for value in thresholds_to_evaluate:
    if not any(np.isclose(value, existing) for existing in unique_thresholds):
        unique_thresholds.append(float(value))

threshold_metrics = pd.DataFrame(
    [metric_row(y_test, y_prob, threshold) for threshold in unique_thresholds]
)
threshold_metrics.to_csv(
    BASE_DIR / "default_decision_tree_test_threshold_metrics.csv",
    index=False,
)

comparison = pd.DataFrame([metrics_050, metrics_best])
comparison.insert(0, "threshold_label", ["Default 0.50", "Exploratory ROC-based"])
comparison.to_csv(
    BASE_DIR / "default_decision_tree_test_performance_comparison.csv",
    index=False,
)

# 17-21. Extract transformed and grouped feature importance.
decision_tree = model.steps[-1][1]
preprocessor = model.steps[0][1]

try:
    transformed_feature_names = preprocessor.get_feature_names_out()
    transformed_feature_names = [
        name.split("__", 1)[1] if "__" in name else name
        for name in transformed_feature_names
    ]
except Exception:
    transformed_feature_names = [
        f"feature_{i}" for i in range(len(decision_tree.feature_importances_))
    ]

feature_importance = pd.DataFrame({
    "transformed_feature": transformed_feature_names,
    "importance": decision_tree.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

feature_importance.insert(0, "rank", np.arange(1, len(feature_importance) + 1))
feature_importance.to_csv(
    BASE_DIR / "default_decision_tree_feature_importance.csv",
    index=False,
)

original_variables = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "Gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

def map_to_original_variable(feature_name):
    for original in original_variables:
        if feature_name == original or feature_name.startswith(original + "_"):
            return original
    return feature_name

feature_importance["original_variable"] = feature_importance["transformed_feature"].apply(
    map_to_original_variable
)

grouped_importance = (
    feature_importance
    .groupby("original_variable", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
grouped_importance.insert(0, "rank", np.arange(1, len(grouped_importance) + 1))
grouped_importance.to_csv(
    BASE_DIR / "default_decision_tree_grouped_feature_importance.csv",
    index=False,
)

print("Default threshold metrics:")
print(metrics_050)
print("\nBest exploratory ROC threshold:")
print({
    "threshold": best_threshold,
    "sensitivity": float(tpr[best_idx]),
    "specificity": float(specificity_values[best_idx]),
    "youden_j": float(youden_j[best_idx]),
})
print("\nBest-threshold metrics:")
print(metrics_best)
print("\nTop 20 transformed feature importances:")
print(feature_importance.head(20))
print("\nGrouped variable importances:")
print(grouped_importance)
