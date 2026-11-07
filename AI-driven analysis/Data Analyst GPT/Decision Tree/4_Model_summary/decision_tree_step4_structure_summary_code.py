
"""
Step 4: Summarize the structure of the trained default decision tree model.
Restrictions followed:
- Uses train_dataset.csv only
- Loads saved default_decision_tree_model.pkl
- Does not use test_dataset.csv
- Does not tune, prune, or evaluate model performance
"""

import pandas as pd
import joblib
from pathlib import Path

DATA_DIR = Path("/mnt/data")
train_path = DATA_DIR / "train_dataset.csv"
model_path = DATA_DIR / "default_decision_tree_model.pkl"
summary_path = DATA_DIR / "default_decision_tree_summary.csv"

# 1. Load training data and saved model pipeline
train_df = pd.read_csv(train_path)
pipeline = joblib.load(model_path)

# 2. Recreate X_train and y_train
target = "stroke"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

# 3. Identify and remove ID-like columns if present
id_like_columns = [
    col for col in X_train.columns
    if col.lower() in ["id", "patient_id", "patientid", "record_id", "recordid"]
    or col.lower().endswith("_id")
]

if id_like_columns:
    X_train = X_train.drop(columns=id_like_columns)

# 4. Extract fitted preprocessing step and decision tree classifier
preprocessor = pipeline.named_steps["preprocess"]
decision_tree = pipeline.named_steps["classifier"]

# 5. Get transformed feature names after one-hot encoding
transformed_feature_names = preprocessor.get_feature_names_out()

# 6. Summarize tree structure
summary_rows = [
    ("Input predictors before preprocessing", X_train.shape[1]),
    ("Transformed features after one-hot encoding", len(transformed_feature_names)),
    ("Total tree nodes", decision_tree.tree_.node_count),
    ("Leaf nodes", decision_tree.tree_.n_leaves),
    ("Maximum tree depth", decision_tree.tree_.max_depth),
    ("Number of classes", decision_tree.n_classes_),
]

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
summary_df.to_csv(summary_path, index=False)

print(summary_df)
print(f"Saved summary to: {summary_path}")
