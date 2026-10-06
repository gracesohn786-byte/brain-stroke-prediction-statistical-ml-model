
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, balanced_accuracy_score
)

MODEL_PATH = "tuned_random_forest_model.pkl"
TEST_PATH = "test_dataset.csv"
TARGET = "stroke"
SELECTED_THRESHOLD = 0.350000

model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

id_like_cols = [c for c in test_df.columns if c.lower() in ["id", "patient_id", "patientid", "identifier"]]
X_test = test_df.drop(columns=[TARGET] + id_like_cols)
y_test = test_df[TARGET].astype(int)

stroke_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, stroke_probability)
pr_auc = average_precision_score(y_test, stroke_probability)

def evaluate_threshold(threshold):
    pred = (stroke_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, pred),
        "sensitivity_recall": recall_score(y_test, pred, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

metrics = pd.DataFrame([
    {"threshold_label": "Default 0.50", **evaluate_threshold(0.50)},
    {"threshold_label": "Training-selected 0.35", **evaluate_threshold(SELECTED_THRESHOLD)}
])
metrics.to_csv("tuned_random_forest_test_metrics.csv", index=False)

thresholds = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
threshold_comparison = pd.DataFrame([evaluate_threshold(t) for t in thresholds])
threshold_comparison.to_csv("tuned_random_forest_test_threshold_comparison.csv", index=False)

predictions = pd.DataFrame({
    "actual_stroke": y_test,
    "predicted_stroke_probability": stroke_probability,
    "predicted_class_threshold_0_50": (stroke_probability >= 0.50).astype(int),
    "predicted_class_threshold_0_35": (stroke_probability >= SELECTED_THRESHOLD).astype(int)
})
predictions.to_csv("tuned_random_forest_test_predictions.csv", index=False)

fpr, tpr, _ = roc_curve(y_test, stroke_probability)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="No-discrimination line")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("Tuned Random Forest Test ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("tuned_random_forest_test_roc_curve.png", dpi=200)
plt.close()

precision, recall, _ = precision_recall_curve(y_test, stroke_probability)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision / PPV")
plt.title("Tuned Random Forest Test Precision-Recall Curve")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("tuned_random_forest_test_precision_recall_curve.png", dpi=200)
plt.close()
