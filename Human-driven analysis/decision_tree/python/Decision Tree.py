# =========================================================
# DECISION TREE MODEL FOR STROKE PREDICTION
# =========================================================

# -------------------- LIBRARIES --------------------
# Install if needed:
# pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
import time 

import pandas as pd
#import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)

# =========================================================
# READ DATA
# =========================================================

# Replace with your file path
mydata = pd.read_excel(r"C:\Users\Lenovo\OneDrive\문서\ATP\Human-driven analysis\data\full_data_real.xlsx")

# View first rows
print(mydata.head())

# Check structure
print(mydata.info())

# =========================================================
# CONVERT CATEGORICAL VARIABLES
# =========================================================

categorical_cols = [
    "Gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
    "hypertension",
    "heart_disease"
]

# Convert to category datatype
for col in categorical_cols:
    mydata[col] = mydata[col].astype("category")

# =========================================================
# ONE-HOT ENCODING
# =========================================================

mydata_encoded = pd.get_dummies(
    mydata,
    columns=categorical_cols,
    drop_first=True
)

# =========================================================
# LOAD TRAIN INDICES FROM R
# =========================================================

# Read indices exported from R
train_indices = pd.read_csv(r"C:\Users\Lenovo\OneDrive\문서\ATP\Human-driven analysis\log_reg\train_indices.csv")

# Convert first column to list
train_indices = train_indices.iloc[:, 0].tolist()

# R starts at 1, Python starts at 0
train_indices = [i - 1 for i in train_indices]

# =========================================================
# CREATE TRAIN / TEST DATASETS
# =========================================================

train_data = mydata_encoded.iloc[train_indices]

test_data = mydata_encoded.drop(train_indices)

# =========================================================
# FEATURES AND TARGET
# =========================================================

X_train = train_data.drop("stroke", axis=1)
y_train = train_data["stroke"]

X_test = test_data.drop("stroke", axis=1)
y_test = test_data["stroke"]

# Check dataset sizes
print("\nTraining rows:", len(train_data))
print("Testing rows:", len(test_data))

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =========================================================
# TRAIN DECISION TREE
# =========================================================
start_time = time.time()

tree_model = DecisionTreeClassifier(
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=5,
    ccp_alpha=0.0001,
    #class_weight="balanced",
    random_state=42
)

tree_model.fit(X_train, y_train)

end_time = time.time()

training_time = end_time - start_time

print(f"Training Time: {training_time:.4f} seconds")

summary_df = pd.DataFrame({
    "Metric": [
        "Tree Depth",
        "Number of Nodes",
        "Number of Leaves",
        "Max Depth Parameter",
        "Min Samples Split",
        "Min Samples Leaf",
        "CCP Alpha"
    ],
    "Value": [
        tree_model.get_depth(),
        tree_model.tree_.node_count,
        tree_model.get_n_leaves(),
        tree_model.max_depth,
        tree_model.min_samples_split,
        tree_model.min_samples_leaf,
        tree_model.ccp_alpha
    ]
})

print(summary_df)

from sklearn.model_selection import cross_val_score
import pandas as pd

# Get pruning path
path = tree_model.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = path.ccp_alphas
impurities = path.impurities

cp_table = pd.DataFrame({
    "CCP Alpha": ccp_alphas,
    "Total Impurity": impurities
})

print(cp_table)

rows = []

for alpha in ccp_alphas:

    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        ccp_alpha=alpha,
        random_state=42
    )

    clf.fit(X_train, y_train)

    cv_scores = cross_val_score(
        clf,
        X_train,
        y_train,
        cv=10,
        scoring="accuracy"
    )

    rows.append({
        "CCP Alpha": round(alpha, 6),
        "Number of Leaves": clf.get_n_leaves(),
        "Tree Depth": clf.get_depth(),
        "Train Error": round(1 - clf.score(X_train, y_train), 4),
        "CV Error": round(1 - cv_scores.mean(), 4),
        "CV Std": round(cv_scores.std(), 4)
    })

cp_table = pd.DataFrame(rows)

print(cp_table)
# =========================================================
# MODEL PERFORMANCE
# =========================================================

# Predicted classes
y_pred = tree_model.predict(X_test)

# Predicted probabilities
y_probs = tree_model.predict_proba(X_test)[:, 1]

# =========================================================
# LOWER CLASSIFICATION THRESHOLD
# =========================================================

threshold = 0.0512

# Convert probabilities into class predictions
y_pred_threshold = (y_probs >= threshold).astype(int)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred_threshold)

print("\nTest Accuracy:", round(accuracy, 4))

# =========================================================
# CONFUSION MATRIX
# =========================================================

conf_mat = confusion_matrix(y_test, y_pred_threshold)

print("\nConfusion Matrix:")
print(conf_mat)

# Extract values
TN = conf_mat[0, 0]
FP = conf_mat[0, 1]
FN = conf_mat[1, 0]
TP = conf_mat[1, 1]

# Sensitivity
sensitivity = TP / (TP + FN)

# Specificity
specificity = TN / (TN + FP)

print("\nSensitivity:", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

auc_value = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(f"ROC Curve (AUC = {auc_value:.3f})")

plt.show()


# =========================================================
# FIND BEST THRESHOLD USING YOUDEN INDEX
# =========================================================

youden_index = tpr - fpr

best_index = youden_index.argmax()

best_threshold = thresholds[best_index]
threshold = thresholds[best_index]
print("\nBest Threshold:", round(best_threshold, 4))
print("Best Sensitivity:", round(tpr[best_index], 4))
print("Best Specificity:", round(1 - fpr[best_index], 4))

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

plt.figure(figsize=(20,10))

plot_tree(
    tree_model,
    feature_names=X_train.columns,
    class_names=["No Stroke", "Stroke"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Stroke Prediction")

plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "Variable": X_train.columns,
    "Importance": tree_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nVariable Importance:")
print(importance_df)

# =========================================================
# FEATURE IMPORTANCE PLOT
# =========================================================

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Variable"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Variables")

plt.title("Decision Tree Variable Importance")

plt.gca().invert_yaxis()

plt.show()

# =========================================================
# CONFUSION MATRIX HEATMAP
# =========================================================

cm_df = pd.DataFrame(
    conf_mat,
    index=["Actual No Stroke", "Actual Stroke"],
    columns=["Predicted No Stroke", "Predicted Stroke"]
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix for Decision Tree")

plt.show()

# =========================================================
# TRAIN VS TEST ACCURACY
# =========================================================

# Training predictions
train_probs = tree_model.predict_proba(X_train)[:, 1]

train_pred = (train_probs >= threshold).astype(int)

train_accuracy = accuracy_score(y_train, train_pred)

print("\nTrain Accuracy:", round(train_accuracy, 4))
print("Test Accuracy:", round(accuracy, 4))

# =========================================================
# EXPORT FEATURE IMPORTANCE
# =========================================================

importance_df.to_csv(
    "decision_tree_importance.csv",
    index=False
)

print("\nFeature importance file saved.")