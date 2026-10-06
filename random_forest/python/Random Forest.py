# =========================================================
# RANDOM FOREST MODEL FOR STROKE PREDICTION
# =========================================================

# -------------------- LIBRARIES --------------------
# Install if needed:
# pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
import time

import pandas as pd
#import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
mydata = pd.read_excel(
    r"C:\Users\Lenovo\OneDrive\문서\Data Visulaization\ATP\Stroke Excel Datasets\full_data_real.xlsx"
)
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
train_indices = pd.read_csv(r"C:\Users\Lenovo\OneDrive\문서\Data Visulaization\ATP\Stroke Excel Datasets\train_indices.csv")

# Convert first column to list
train_indices = train_indices.iloc[:, 0].tolist()

# R starts at 1, Python starts at 0
train_indices = [i - 1 for i in train_indices]

# =========================================================
# CREATE TRAIN / TEST DATASETS
# =========================================================

train_data = mydata_encoded.iloc[train_indices]

test_data = mydata_encoded.drop(train_indices)

summary_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Observations": [len(train_data), len(test_data)],
    "Stroke Cases": [
        train_data["stroke"].sum(),
        test_data["stroke"].sum()
    ],
    "Stroke Rate (%)": [
        train_data["stroke"].mean() * 100,
        test_data["stroke"].mean() * 100
    ]
})

print(summary_df)

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
# TRAIN RANDOM FOREST
# =========================================================
# Start timer
start_time = time.time()

forest_model = RandomForestClassifier(
    
    n_estimators=100,
    random_state=42,
    max_depth=6,
    
    class_weight="balanced"
)

forest_model.fit(X_train, y_train)

# End timer
end_time = time.time()

training_time = end_time - start_time

print(f"Training Time: {training_time:.4f} seconds")
# =========================================================
# MODEL PERFORMANCE
# =========================================================

# Predictions
y_pred = forest_model.predict(X_test)

# Predicted probabilities
y_probs = forest_model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy, 4))

# =========================================================
# CONFUSION MATRIX
# =========================================================

conf_mat = confusion_matrix(y_test, y_pred)

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

# Best threshold
youden = tpr - fpr
best_idx = youden.argmax()

best_threshold = thresholds[best_idx]

print(best_threshold)
print(tpr[best_idx])
print(1 - fpr[best_idx])

y_pred = (y_probs >= best_threshold).astype(int)
# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "Variable": X_train.columns,
    "Importance": forest_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nVariable Importance:")
print(importance_df)

# Plot importance
plt.figure(figsize=(10,7))

plt.barh(
    importance_df["Variable"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Variables")

plt.title("Random Forest Variable Importance")

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

plt.title("Confusion Matrix for Random Forest")

plt.show()

# =========================================================
# TRAIN VS TEST ACCURACY
# =========================================================

# Training predictions
train_pred = forest_model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_pred)

print("\nTrain Accuracy:", round(train_accuracy, 4))
print("Test Accuracy:", round(accuracy, 4))

# =========================================================
# OOB SCORE (OUT-OF-BAG ERROR)
# =========================================================

forest_oob = RandomForestClassifier(
    
    n_estimators=100,
    random_state=42,
    
    oob_score=True,
    
    class_weight="balanced"
)

forest_oob.fit(X_train, y_train)

print("\nOOB Score:", round(forest_oob.oob_score_, 4))

# =========================================================
# OPTIONAL: TEST DIFFERENT TREE COUNTS
# =========================================================

tree_values = [10, 50, 100, 200]

accuracy_list = []

for n in tree_values:
    
    temp_model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        class_weight="balanced"
    )
    
    temp_model.fit(X_train, y_train)
    
    temp_pred = temp_model.predict(X_test)
    
    temp_acc = accuracy_score(y_test, temp_pred)
    
    accuracy_list.append(temp_acc)

# Plot results
plt.figure(figsize=(7,5))

plt.plot(tree_values, accuracy_list, marker="o")

plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")

plt.title("Random Forest Accuracy vs Number of Trees")

plt.show()

# =========================================================
# OPTIONAL: SAVE FEATURE IMPORTANCE
# =========================================================

importance_df.to_csv(
    "random_forest_importance.csv",
    index=False
)

print("\nFeature importance file saved.")