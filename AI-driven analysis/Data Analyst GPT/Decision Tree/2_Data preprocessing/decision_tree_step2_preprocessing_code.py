"""
Decision Tree Analysis - Step 2
Prepare feature matrix, target variable, and preprocessing pipeline for an initial
default decision tree model for brain stroke prediction.

Restrictions followed:
- Uses training dataset only.
- Does not use test_dataset.csv.
- Does not train a decision tree.
- Does not tune, prune, balance, or evaluate a model.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 1. Load training dataset
train_path = "train_dataset.csv"
df_train = pd.read_csv(train_path)

# 2. Separate features and target
target = "stroke"
if target not in df_train.columns:
    raise ValueError("Target variable 'stroke' was not found in the training dataset.")

X_train = df_train.drop(columns=[target])
y_train = df_train[target]

# 3. Identify and remove ID-like columns, if present
id_like_columns = [
    col for col in X_train.columns
    if col.lower() in ["id", "patient_id", "patientid", "record_id", "recordid"]
    or col.lower().endswith("_id")
]

if id_like_columns:
    X_train = X_train.drop(columns=id_like_columns)

# 4. Define predictor groups
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

# Keep only columns present in the loaded training dataset
numerical_predictors = [col for col in numerical_predictors if col in X_train.columns]
categorical_predictors = [col for col in categorical_predictors if col in X_train.columns]

# 5. Build preprocessing pipeline
# Numerical variables: median imputation only.
# Categorical variables: most-frequent imputation + one-hot encoding.
# No scaling is used because decision trees do not require feature scaling.
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_predictors),
        ("cat", categorical_pipeline, categorical_predictors),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# 8. Fit preprocessing only on training features
X_train_transformed = preprocessor.fit_transform(X_train)

# 9-10. Report transformed feature count and feature names
feature_names = preprocessor.get_feature_names_out()

print("Training dataset shape:", df_train.shape)
print("Features removed:", id_like_columns if id_like_columns else "None")
print("Number of original predictors before preprocessing:", X_train.shape[1])
print("Number of transformed features after preprocessing:", X_train_transformed.shape[1])
print("Final numerical predictors:", numerical_predictors)
print("Final categorical predictors:", categorical_predictors)
print("Transformed feature names:")
for name in feature_names:
    print("-", name)
