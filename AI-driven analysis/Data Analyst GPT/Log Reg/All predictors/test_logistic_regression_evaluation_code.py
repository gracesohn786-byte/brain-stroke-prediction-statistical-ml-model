"""
Evaluate a saved tuned logistic regression stroke prediction model on an untouched test dataset.

Important:
- This script does NOT retrain the model.
- This script does NOT refit preprocessing steps.
- Threshold selection using the test ROC curve is exploratory and should not be treated
  as final publication-quality model selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
)

# ---------------------------------------------------------------------
# 1. Load untouched test dataset and saved fitted pipeline
# ---------------------------------------------------------------------
TEST_PATH = "test_dataset.csv"
MODEL_PATH = "logistic_regression_stroke_model.pkl"

test_df = pd.read_csv(TEST_PATH)
pipeline = joblib.load(MODEL_PATH)

TARGET = "stroke"

print("Test dataset shape:", test_df.shape)
print("\nStroke target distribution:")
print(test_df[TARGET].value_counts().sort_index())
print("\nStroke target distribution (%):")
print((test_df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))

# Optional model audit: inspect the saved fitted estimator settings.
try:
    fitted_lr = pipeline.named_steps["model"]
    print("\nSaved logistic regression settings:")
    print("C:", fitted_lr.C)
    print("class_weight:", fitted_lr.class_weight)
except Exception:
    print("\nCould not inspect logistic regression settings from the saved pipeline.")

# ---------------------------------------------------------------------
# 2. Separate features and target
# ---------------------------------------------------------------------
y_test = test_df[TARGET].astype(int)
X_test = test_df.drop(columns=[TARGET])

# Remove an ID column from features if present.
# Patient ID is retained for the output prediction file.
id_columns = [col for col in X_test.columns if col.lower() == "id" or col.lower().endswith("_id")]
if id_columns:
    patient_id = X_test[id_columns[0]].copy()
    X_test_model = X_test.drop(columns=id_columns)
else:
    patient_id = pd.Series(np.arange(1, len(test_df) + 1), name="patient_id")
    X_test_model = X_test.copy()

# ---------------------------------------------------------------------
# 3. Predict stroke probabilities with the saved fitted pipeline
# ---------------------------------------------------------------------
# This calls the already-fitted preprocessing steps and logistic model.
y_proba = pipeline.predict_proba(X_test_model)[:, 1]

# ---------------------------------------------------------------------
# 4. ROC curve, ROC-AUC, and exploratory Youden J threshold selection
# ---------------------------------------------------------------------
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

# Youden's J = sensitivity + specificity - 1 = TPR - FPR
youden_j = tpr - fpr
best_idx = int(np.argmax(youden_j))
selected_threshold = float(roc_thresholds[best_idx])
selected_sensitivity = float(tpr[best_idx])
selected_specificity = float(1 - fpr[best_idx])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="No-discrimination line")
plt.scatter(
    fpr[best_idx],
    tpr[best_idx],
    s=50,
    label=f"Youden J threshold = {selected_threshold:.3f}",
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("Test ROC Curve - Tuned Balanced Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("test_roc_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 5. Precision-recall curve and PR-AUC / Average Precision
# ---------------------------------------------------------------------
average_precision = average_precision_score(y_test, y_proba)
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f"PR curve (AP = {average_precision:.3f})")
plt.axhline(
    y=y_test.mean(),
    linestyle="--",
    label=f"Baseline stroke prevalence = {y_test.mean():.3f}",
)
plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision / PPV")
plt.title("Test Precision-Recall Curve - Tuned Balanced Logistic Regression")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("test_precision_recall_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 6. Metric helper for a chosen classification threshold
# ---------------------------------------------------------------------
def evaluate_threshold(threshold: float) -> dict:
    """Calculate classification metrics at a given probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_ppv": precision_score(y_test, y_pred, zero_division=0),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "negative_predictive_value": npv,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# ---------------------------------------------------------------------
# 7. Evaluate default threshold and selected ROC-based threshold
# ---------------------------------------------------------------------
default_metrics = evaluate_threshold(0.50)
selected_metrics = evaluate_threshold(selected_threshold)

primary_summary = pd.DataFrame([default_metrics, selected_metrics])
primary_summary.insert(0, "threshold_label", ["Default 0.50", "ROC Youden J threshold"])
primary_summary["test_roc_auc"] = roc_auc
primary_summary["test_pr_auc_average_precision"] = average_precision
primary_summary["row_type"] = "primary_threshold_summary"

print("\nTest ROC-AUC:", round(roc_auc, 4))
print("Test PR-AUC / Average Precision:", round(average_precision, 4))
print("\nExploratory selected ROC-based threshold:", round(selected_threshold, 4))
print("Sensitivity at selected threshold:", round(selected_sensitivity, 4))
print("Specificity at selected threshold:", round(selected_specificity, 4))
print("\nPrimary threshold performance:")
print(primary_summary)

# ---------------------------------------------------------------------
# 8. Threshold comparison table
# ---------------------------------------------------------------------
candidate_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, selected_threshold]

threshold_rows = []
seen_thresholds = []
for threshold in candidate_thresholds:
    # Avoid duplicate rows if selected threshold equals one of the fixed thresholds.
    if any(abs(threshold - existing) < 1e-10 for existing in seen_thresholds):
        continue

    seen_thresholds.append(threshold)
    row = evaluate_threshold(threshold)
    row["threshold_label"] = (
        "Selected ROC Youden J"
        if abs(threshold - selected_threshold) < 1e-10
        else f"{threshold:.2f}"
    )
    threshold_rows.append(row)

threshold_comparison = pd.DataFrame(threshold_rows)
threshold_comparison = threshold_comparison[
    [
        "threshold_label",
        "threshold",
        "accuracy",
        "precision_ppv",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "negative_predictive_value",
        "balanced_accuracy",
        "fp",
        "fn",
        "tn",
        "tp",
    ]
]
threshold_comparison["row_type"] = "threshold_comparison"

print("\nThreshold comparison:")
print(threshold_comparison)

# ---------------------------------------------------------------------
# 9. Save patient-level predictions
# ---------------------------------------------------------------------
predictions = pd.DataFrame(
    {
        "patient_id": patient_id.values,
        "actual_stroke": y_test.values,
        "predicted_stroke_probability": y_proba,
        "predicted_class_threshold_0_50": (y_proba >= 0.50).astype(int),
        "predicted_class_selected_roc_threshold": (y_proba >= selected_threshold).astype(int),
    }
)

# Use the original ID column name if one existed.
if id_columns:
    predictions = predictions.rename(columns={"patient_id": id_columns[0]})

predictions.to_csv("test_predictions_logistic_regression.csv", index=False)

# ---------------------------------------------------------------------
# 10. Save metrics CSV
# ---------------------------------------------------------------------
metrics_output = pd.concat(
    [primary_summary, threshold_comparison],
    ignore_index=True,
    sort=False,
)

ordered_columns = [
    "row_type",
    "threshold_label",
    "threshold",
    "accuracy",
    "precision_ppv",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "negative_predictive_value",
    "balanced_accuracy",
    "test_roc_auc",
    "test_pr_auc_average_precision",
    "fp",
    "fn",
    "tn",
    "tp",
]

metrics_output[ordered_columns].to_csv("test_logistic_regression_metrics.csv", index=False)

print("\nSaved outputs:")
print("- test_roc_curve.png")
print("- test_precision_recall_curve.png")
print("- test_predictions_logistic_regression.csv")
print("- test_logistic_regression_metrics.csv")
