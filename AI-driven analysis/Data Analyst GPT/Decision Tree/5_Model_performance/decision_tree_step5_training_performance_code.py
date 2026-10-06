# Apparent training performance for default decision tree stroke model
# Uses only train_dataset.csv and default_decision_tree_model.pkl.
# Does not use testing data, tune, prune, rebalance, or select thresholds.

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

TRAIN_PATH = "train_dataset.csv"
MODEL_PATH = "default_decision_tree_model.pkl"
TARGET = "stroke"

train_df = pd.read_csv(TRAIN_PATH)
model = joblib.load(MODEL_PATH)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

id_like_cols = [
    col for col in X_train.columns
    if col.lower() in {"id", "patient_id", "patientid", "subject_id", "record_id"}
    or col.lower().endswith("_id")
]
if id_like_cols:
    X_train = X_train.drop(columns=id_like_cols)

y_pred_train = model.predict(X_train)
y_prob_train = model.predict_proba(X_train)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train, labels=[0, 1]).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

metrics = {
    "dataset": "training",
    "performance_type": "apparent training performance",
    "records": len(y_train),
    "stroke_cases": int((y_train == 1).sum()),
    "no_stroke_cases": int((y_train == 0).sum()),
    "accuracy": accuracy_score(y_train, y_pred_train),
    "precision_ppv": precision_score(y_train, y_pred_train, zero_division=0),
    "recall_sensitivity": recall_score(y_train, y_pred_train, zero_division=0),
    "specificity": specificity,
    "f1_score": f1_score(y_train, y_pred_train, zero_division=0),
    "roc_auc": roc_auc_score(y_train, y_prob_train),
    "pr_auc_average_precision": average_precision_score(y_train, y_prob_train),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("default_decision_tree_training_metrics.csv", index=False)
print(metrics_df.T)
