import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.tree import plot_tree, export_text

# ---------------------------------------------------------------------
# Tuned/pruned decision tree test evaluation
# Important: this script loads the saved model and evaluates it on the
# untouched test dataset. It does not retrain, retune, prune, rebalance,
# or refit preprocessing steps.
# ---------------------------------------------------------------------

TEST_PATH = "test_dataset.csv"
MODEL_PATH = "tuned_sensitivity_decision_tree_model.pkl"
TARGET = "stroke"

df_test = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)

id_like_cols = [
    c for c in df_test.columns
    if c.lower() in ["id", "patient_id", "patientid", "patient id"]
]

X_test = df_test.drop(columns=[TARGET])
y_test = df_test[TARGET].astype(int)

if id_like_cols:
    X_test = X_test.drop(columns=id_like_cols, errors="ignore")

# Predicted probabilities for stroke = 1
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

def metrics_at_threshold(threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity_recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

# Default threshold
default_metrics = metrics_at_threshold(0.50)

# Exploratory ROC-based threshold using Youden's J
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
specificity_values = 1 - fpr
youden_j = tpr + specificity_values - 1
best_idx = np.argmax(youden_j)
best_threshold = roc_thresholds[best_idx]

best_threshold_metrics = metrics_at_threshold(best_threshold)

# Save ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="No discrimination")
plt.scatter(fpr[best_idx], tpr[best_idx], label=f"Youden threshold = {best_threshold:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("Tuned Decision Tree Test ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("tuned_decision_tree_test_roc_curve.png", dpi=200)
plt.close()

# Save precision-recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR curve (AP = {pr_auc:.3f})")
plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision / PPV")
plt.title("Tuned Decision Tree Test Precision-Recall Curve")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("tuned_decision_tree_test_precision_recall_curve.png", dpi=200)
plt.close()

# Threshold comparison
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, float(best_threshold)]
thresholds = list(dict.fromkeys([round(float(t), 10) for t in thresholds]))
threshold_comparison = pd.DataFrame([metrics_at_threshold(t) for t in thresholds])
threshold_comparison.to_csv("tuned_decision_tree_test_threshold_metrics.csv", index=False)

# Main test metrics
metrics_df = pd.DataFrame([
    {
        **default_metrics,
        "threshold_label": "0.50",
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
    },
    {
        **best_threshold_metrics,
        "threshold_label": "exploratory_roc_youden",
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
        "youden_j": youden_j[best_idx],
        "roc_threshold_sensitivity": tpr[best_idx],
        "roc_threshold_specificity": specificity_values[best_idx],
    },
])
metrics_df.to_csv("tuned_decision_tree_test_metrics.csv", index=False)

# Predictions file
predictions = pd.DataFrame({
    "actual_stroke": y_test.values,
    "predicted_stroke_probability": y_prob,
    "predicted_class_threshold_0_50": (y_prob >= 0.50).astype(int),
    "predicted_class_exploratory_roc_threshold": (y_prob >= best_threshold).astype(int),
})

for c in reversed(id_like_cols):
    predictions.insert(0, c, df_test[c].values)

predictions.to_csv("tuned_decision_tree_test_predictions.csv", index=False)

# Feature importance
preprocess = model.named_steps["preprocess"]
tree = model.named_steps["model"]
feature_names = preprocess.get_feature_names_out()
feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

feature_importance = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": tree.feature_importances_,
    })
    .sort_values("importance", ascending=False)
)
feature_importance.to_csv("tuned_decision_tree_test_feature_importance.csv", index=False)

def group_feature(feature):
    numeric = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    categorical = ["Gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for v in numeric:
        if feature == v or feature.startswith(v + "_"):
            return v
    for v in categorical:
        if feature == v or feature.startswith(v + "_"):
            return v
    return feature

feature_importance["original_variable"] = feature_importance["feature"].apply(group_feature)
grouped_importance = (
    feature_importance.groupby("original_variable", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)
grouped_importance.to_csv("tuned_decision_tree_test_grouped_feature_importance.csv", index=False)

# Tree plot and rules
plt.figure(figsize=(14, 8))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["No Stroke", "Stroke"],
    filled=True,
    rounded=True,
    proportion=True,
)
plt.title("Tuned Class-Weighted Decision Tree")
plt.tight_layout()
plt.savefig("tuned_decision_tree_plot.png", dpi=200)
plt.close()

rules = export_text(tree, feature_names=feature_names, decimals=3)
with open("tuned_decision_tree_rules.txt", "w") as f:
    f.write(rules)