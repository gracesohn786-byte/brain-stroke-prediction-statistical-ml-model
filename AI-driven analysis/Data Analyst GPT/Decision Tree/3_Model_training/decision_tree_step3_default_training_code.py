"""
Step 3: Train an initial default decision tree model for brain stroke prediction.

Restrictions followed:
- Uses only train_dataset.csv
- Does not use test_dataset.csv
- Does not tune hyperparameters
- Does not prune the tree
- Does not use class weighting
- Does not use SMOTE or resampling
- Uses default DecisionTreeClassifier settings except random_state=42
"""

import time
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# 1. Load training data
train_df = pd.read_csv("train_dataset.csv")

# 2. Separate features and target
target = "stroke"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

# 3. Remove ID-like columns if present
id_like_columns = [
    col for col in X_train.columns
    if col.lower() == "id"
    or col.lower().endswith("_id")
    or col.lower().startswith("id_")
    or col.lower() in ["patient_id", "patientid"]
]

if id_like_columns:
    X_train = X_train.drop(columns=id_like_columns)

# 4. Define predictors
numerical_predictors = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

categorical_predictors = [
    "Gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# Keep only predictors that exist in the loaded training file
numerical_predictors = [col for col in numerical_predictors if col in X_train.columns]
categorical_predictors = [col for col in categorical_predictors if col in X_train.columns]

# 5. Build preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_predictors),
        ("cat", categorical_transformer, categorical_predictors),
    ],
    remainder="drop",
)

# 6. Build default decision tree pipeline
decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]
)

# 7. Record training time and fit model
start_time = time.perf_counter()
decision_tree_pipeline.fit(X_train, y_train)
end_time = time.perf_counter()

elapsed_training_time = end_time - start_time

# 8. Save trained pipeline
joblib.dump(decision_tree_pipeline, "default_decision_tree_model.pkl")

# 9. Report model configuration
tree = decision_tree_pipeline.named_steps["classifier"]

print("Model trained successfully.")
print(f"Training records: {X_train.shape[0]}")
print(f"Predictors used before preprocessing: {X_train.shape[1]}")
print(f"Features removed: {id_like_columns if id_like_columns else 'None'}")
print(f"Training time seconds: {elapsed_training_time:.6f}")

print("\nDefault DecisionTreeClassifier configuration:")
print(f"criterion: {tree.criterion}")
print(f"splitter: {tree.splitter}")
print(f"max_depth: {tree.max_depth}")
print(f"min_samples_split: {tree.min_samples_split}")
print(f"min_samples_leaf: {tree.min_samples_leaf}")
print(f"max_features: {tree.max_features}")
print(f"class_weight: {tree.class_weight}")
print(f"random_state: {tree.random_state}")
