import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, balanced_accuracy_score
)

# Paths
MODEL_PATH = "initial_random_forest_model.pkl"
TEST_PATH = "test_dataset.csv"

# Load model and test data
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

target = "stroke"
id_like_columns = [
    col for col in test_df.columns
    if col.lower() in ["id", "patient_id", "patientid", "patient id"]
]

X_test = test_df.drop(columns=[target])
y_test = test_df[target].astype(int)

# Remove ID-like columns consistently if present.
if id_like_columns:
    X_test = X_test.drop(columns=[col for col in id_like_columns if col in X_test.columns])

# Predicted probabilities for positive class: stroke = 1
y_prob = model.predict_proba(X_test)[:, 1]

# Overall curve metrics
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

# ROC curve and exploratory Youden threshold
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
youden_j = tpr - fpr
finite_thresholds = np.isfinite(roc_thresholds)
best_idx = np.argmax(np.where(finite_thresholds, youden_j, -np.inf))
best_threshold = roc_thresholds[best_idx]

def calculate_metrics(threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity_recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "precision_ppv": precision_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# Performance at threshold 0.50 and exploratory ROC threshold
metrics_050 = calculate_metrics(0.50)
metrics_best = calculate_metrics(best_threshold)

metrics_df = pd.DataFrame([
    {"threshold_label": "Default 0.50", **metrics_050},
    {"threshold_label": "Exploratory ROC-based", **metrics_best},
])
metrics_df.to_csv("random_forest_test_metrics.csv", index=False)

# Threshold comparison table
threshold_rows = []
for label, threshold in [
    ("0.10", 0.10),
    ("0.20", 0.20),
    ("0.30", 0.30),
    ("0.40", 0.40),
    ("0.50", 0.50),
    ("Exploratory ROC-based", float(best_threshold)),
]:
    threshold_rows.append({"threshold_label": label, **calculate_metrics(threshold)})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv("random_forest_test_threshold_comparison.csv", index=False)

# Test predictions
predictions = pd.DataFrame()
for col in id_like_columns:
    if col in test_df.columns:
        predictions[col] = test_df[col]

predictions["actual_stroke"] = y_test.values
predictions["predicted_stroke_probability"] = y_prob
predictions["predicted_class_threshold_0_50"] = (y_prob >= 0.50).astype(int)
predictions["predicted_class_exploratory_roc_threshold"] = (y_prob >= best_threshold).astype(int)
predictions.to_csv("random_forest_test_predictions.csv", index=False)

# ROC plot
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="No-discrimination line")
plt.scatter(fpr[best_idx], tpr[best_idx],
            label=f"Youden threshold = {best_threshold:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("Random Forest Test ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("random_forest_test_roc_curve.png", dpi=200)
plt.close()

# Precision-recall plot
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR curve (AP = {pr_auc:.4f})")
plt.axhline(y=y_test.mean(), linestyle="--",
            label=f"Baseline prevalence = {y_test.mean():.4f}")
plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision / PPV")
plt.title("Random Forest Test Precision-Recall Curve")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("random_forest_test_precision_recall_curve.png", dpi=200)
plt.close()

print("ROC-AUC:", roc_auc)
print("PR-AUC / Average Precision:", pr_auc)
print("Exploratory Youden threshold:", best_threshold)
print(metrics_df)
