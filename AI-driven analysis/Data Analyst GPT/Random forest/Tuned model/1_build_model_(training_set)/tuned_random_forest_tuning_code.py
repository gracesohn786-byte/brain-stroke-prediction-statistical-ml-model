import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score
)

# Training-only random forest tuning for stroke prediction.
# The test dataset is intentionally not loaded or used in this step.

base = Path("/mnt/data")
df = pd.read_csv(base / "train_dataset.csv")
target = "stroke"

id_like = [c for c in df.columns if c.lower() in ["id", "patient_id", "patientid", "record_id", "recordid"]]
X = df.drop(columns=[target] + id_like)
y = df[target].astype(int).values

numerical_predictors = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
categorical_predictors = ["Gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numerical_predictors = [c for c in numerical_predictors if c in X.columns]
categorical_predictors = [c for c in categorical_predictors if c in X.columns]

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_predictors),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_predictors)
        ],
        remainder="drop"
    )

def metrics_from_prob(y_true, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predicted),
        "precision_ppv": precision_score(y_true, predicted, zero_division=0),
        "sensitivity_recall": recall_score(y_true, predicted, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc_avg_precision": average_precision_score(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# Candidate models intentionally cover baseline, class-weighting, balanced-subsample,
# limited depth, larger leaf sizes, and custom stroke-class weighting.
candidate_configs = [
    ("Baseline RF", dict(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", class_weight=None)),
    ("Class-weighted RF", dict(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", class_weight="balanced")),
    ("Balanced-subsample RF", dict(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample")),
    ("Limited-depth balanced RF", dict(n_estimators=100, max_depth=4, min_samples_split=10, min_samples_leaf=5, max_features="sqrt", class_weight="balanced")),
    ("Large-leaf balanced RF", dict(n_estimators=100, max_depth=6, min_samples_split=20, min_samples_leaf=20, max_features="sqrt", class_weight="balanced")),
    ("Custom weight RF 1:10", dict(n_estimators=100, max_depth=6, min_samples_split=10, min_samples_leaf=5, max_features="sqrt", class_weight={0: 1, 1: 10})),
    ("Shallow custom RF 1:10", dict(n_estimators=100, max_depth=3, min_samples_split=10, min_samples_leaf=10, max_features="sqrt", class_weight={0: 1, 1: 10})),
]

# Use stratified 5-fold cross-validation to generate out-of-fold probabilities.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_oof_probabilities(params):
    probability = np.zeros(len(y))
    for train_idx, valid_idx in cv.split(X, y):
        preprocessor = make_preprocessor()
        X_train_fold = preprocessor.fit_transform(X.iloc[train_idx])
        X_valid_fold = preprocessor.transform(X.iloc[valid_idx])
        model = RandomForestClassifier(random_state=42, bootstrap=True, n_jobs=-1, **params)
        model.fit(X_train_fold, y[train_idx])
        probability[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
    return probability

def build_threshold_table(y_true, probability):
    thresholds = np.unique(np.concatenate([
        np.linspace(0, 0.5, 101),
        np.quantile(probability, np.linspace(0, 1, 101))
    ]))
    return pd.DataFrame([metrics_from_prob(y_true, probability, threshold) for threshold in thresholds])

def choose_threshold(threshold_table, minimum_sensitivity=0.80):
    eligible = threshold_table[threshold_table["sensitivity_recall"] >= minimum_sensitivity].copy()
    if len(eligible):
        return eligible.sort_values(
            ["specificity", "precision_ppv", "f1_score", "threshold"],
            ascending=[False, False, False, False]
        ).iloc[0]
    threshold_table = threshold_table.copy()
    threshold_table["youden_j"] = threshold_table["sensitivity_recall"] + threshold_table["specificity"] - 1
    return threshold_table.sort_values(["youden_j", "sensitivity_recall"], ascending=False).iloc[0]

results = []
for model_name, params in candidate_configs:
    probability = get_oof_probabilities(params)
    threshold_table = build_threshold_table(y, probability)
    selected_threshold = choose_threshold(threshold_table, minimum_sensitivity=0.80)["threshold"]
    metrics = metrics_from_prob(y, probability, selected_threshold)
    metrics.update({"model": model_name, "selected_threshold": selected_threshold})
    metrics.update({f"param_{key}": str(value) for key, value in params.items()})
    results.append(metrics)

comparison = pd.DataFrame(results)

# Select the model that reaches sensitivity >=0.80 while minimizing false positives,
# with specificity/PR-AUC used as secondary considerations.
eligible = comparison[comparison["sensitivity_recall"] >= 0.80].copy()
selected = eligible.sort_values(
    ["fp", "specificity", "pr_auc_avg_precision"],
    ascending=[True, False, False]
).iloc[0]

selected_params = dict(
    n_estimators=100,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced"
)
selected_threshold = float(selected["selected_threshold"])

final_pipeline = Pipeline(steps=[
    ("preprocess", make_preprocessor()),
    ("rf", RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        n_jobs=-1,
        **selected_params
    ))
])

start = time.perf_counter()
final_pipeline.fit(X, y)
training_time = time.perf_counter() - start

joblib.dump(final_pipeline, base / "tuned_random_forest_model.pkl")