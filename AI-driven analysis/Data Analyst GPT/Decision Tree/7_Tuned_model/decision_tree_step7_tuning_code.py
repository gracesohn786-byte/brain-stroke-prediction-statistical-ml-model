
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score

df = pd.read_csv("train_dataset.csv")
X_train = df.drop(columns=["stroke"])
y_train = df["stroke"].astype(int)

id_like_cols = [c for c in X_train.columns if c.lower() in ["id", "patient_id", "patientid", "patient id"] or c.lower().endswith("_id")]
if id_like_cols:
    X_train = X_train.drop(columns=id_like_cols)

num_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
cat_cols = ["Gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

base_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_common = {
    "model__max_depth": [2, 3, 4, 5, 6, 8],
    "model__min_samples_split": [2, 20],
    "model__min_samples_leaf": [1, 20],
    "model__ccp_alpha": [0.0, 0.001, 0.005],
}

grids = {
    "Model A: Pruned Decision Tree": {**param_common, "model__class_weight": [None]},
    "Model B: Class-Weighted Decision Tree": {**param_common, "model__class_weight": ["balanced"]},
    "Model C: Sensitivity-Focused Class-Weighted Decision Tree": {
        **param_common,
        "model__class_weight": [{0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}, {0: 1, 1: 20}]
    },
}

def evaluate_cv(estimator, X, y, cv):
    y_pred = cross_val_predict(estimator, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision_ppv": precision_score(y, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y, y_pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc_average_precision": average_precision_score(y, y_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

results = []
estimators = {}
for model_name, param_grid in grids.items():
    search = GridSearchCV(base_pipe, param_grid, scoring="recall", cv=cv, refit=True)
    search.fit(X_train, y_train)
    metrics = evaluate_cv(search.best_estimator_, X_train, y_train, cv)
    results.append({"model_strategy": model_name, "grid_best_mean_cv_recall": search.best_score_, **metrics, **search.best_params_})
    estimators[model_name] = search.best_estimator_

comparison = pd.DataFrame(results)
comparison.to_csv("tuned_decision_tree_cv_model_comparison.csv", index=False)

selected_model = estimators["Model B: Class-Weighted Decision Tree"]
start = time.perf_counter()
selected_model.fit(X_train, y_train)
training_time = time.perf_counter() - start
joblib.dump(selected_model, "tuned_sensitivity_decision_tree_model.pkl")
