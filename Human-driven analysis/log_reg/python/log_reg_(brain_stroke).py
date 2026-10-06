# =========================================================
# LOGISTIC REGRESSION MODEL FOR STROKE PREDICTION
# =========================================================

# -------------------- LIBRARIES --------------------
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

# =========================================================
# READ DATA
# =========================================================

mydata = pd.read_excel("full_data_real.xlsx")

print(mydata.head())
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

for col in categorical_cols:
    mydata[col] = mydata[col].astype("category")

# =========================================================
# LOAD TRAIN INDICES FROM R
# =========================================================

train_indices = pd.read_csv(r"C:\Users\Lenovo\OneDrive\문서\ATP\Human-driven analysis\log_reg\train_indices.csv")

train_indices = train_indices.iloc[:, 0].tolist()

# Convert R indexing to Python indexing
train_indices = [i - 1 for i in train_indices]

# =========================================================
# TRAIN / TEST DATASETS
# =========================================================

train_data = mydata.iloc[train_indices].copy()

test_data = mydata.drop(train_indices).copy()

print("\nTraining rows:", len(train_data))
print("Testing rows:", len(test_data))

print(train_data.columns.tolist())

# =========================================================
# MAIN LOGISTIC REGRESSION MODEL
# =========================================================

start_time = time.time()

mod_main = smf.logit(
    formula="""
    stroke ~ Gender +
    age +
    hypertension +
    heart_disease +
    ever_married +
    Residence_type +
    avg_glucose_level +
    bmi +
    smoking_status
    """,
    data=train_data
).fit()

training_time = time.time() - start_time

print(f"\nTraining Time: {training_time:.4f} seconds")

# Model summary
print(mod_main.summary())

# =========================================================
# INTERACTION MODEL
# =========================================================

start_time = time.time()

mod_interaction = smf.logit(
    formula="""
    stroke ~ Gender +
    age*bmi +
    heart_disease +
    hypertension +
    bmi*avg_glucose_level
    """,
    data=train_data
).fit()

training_time = time.time() - start_time

print(f"\nTraining Time: {training_time:.4f} seconds")

print(mod_interaction.summary())

# =========================================================
# ODDS RATIOS
# =========================================================

print("\nMAIN MODEL ODDS RATIOS")
odds_ratios_main = np.exp(mod_main.params)
print(odds_ratios_main)

print("\nINTERACTION MODEL ODDS RATIOS")
odds_ratios_interaction = np.exp(mod_interaction.params)
print(odds_ratios_interaction)

# =========================================================
# CONFIDENCE INTERVALS FOR ODDS RATIOS
# =========================================================

print("\nMAIN MODEL CI")
print(np.exp(mod_main.conf_int()))

print("\nINTERACTION MODEL CI")
print(np.exp(mod_interaction.conf_int()))

# =========================================================
# AIC COMPARISON
# =========================================================

print("\nAIC Comparison")
print("Main Model:", mod_main.aic)
print("Interaction Model:", mod_interaction.aic)

# =========================================================
# LIKELIHOOD RATIO TEST
# =========================================================

lr_stat = 2 * (mod_interaction.llf - mod_main.llf)

df_diff = (
    len(mod_interaction.params)
    - len(mod_main.params)
)

from scipy.stats import chi2

p_value = chi2.sf(lr_stat, df_diff)

print("\nLikelihood Ratio Test")
print("Chi-square:", round(lr_stat,4))
print("df:", df_diff)
print("p-value:", p_value)

# =========================================================
# PREDICTED PROBABILITIES
# =========================================================
test_data["test_probs"] = mod_main.predict(test_data)

print(test_data["test_probs"].head())


test_data["test_probs"] = mod_interaction.predict(test_data)

print(test_data["test_probs"].head())

# =========================================================
# AGE VS PROBABILITY
# =========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=test_data["age"],
    y=test_data["test_probs"],
    alpha=0.4
)

plt.title("Predicted Stroke Probability by Age")

plt.xlabel("Age")
plt.ylabel("Predicted Probability")

plt.show()

# =========================================================
# BMI VS PROBABILITY
# =========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=test_data["bmi"],
    y=test_data["test_probs"],
    alpha=0.4
)

plt.title("Predicted Stroke Probability by BMI")

plt.xlabel("BMI")
plt.ylabel("Predicted Probability")

plt.show()

# =========================================================
# THRESHOLD LOOP
# =========================================================

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

results = []

for t in thresholds:

    preds = (
        test_data["test_probs"] >= t
    ).astype(int)

    conf = confusion_matrix(
        test_data["stroke"],
        preds
    )

    TN, FP, FN, TP = conf.ravel()

    accuracy = accuracy_score(
        test_data["stroke"],
        preds
    )

    sensitivity = TP / (TP + FN)

    specificity = TN / (TN + FP)

    results.append([
        t,
        accuracy,
        sensitivity,
        specificity
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Sensitivity",
        "Specificity"
    ]
)

print("\nThreshold Results")
print(results_df)

results_df.to_csv(
    "threshold_results.csv",
    index=False
)

# =========================================================
# SPECIFIC THRESHOLD
# =========================================================

threshold = 0.048734

test_preds = (
    test_data["test_probs"] >= threshold
).astype(int)

accuracy = accuracy_score(
    test_data["stroke"],
    test_preds
)

conf_mat = confusion_matrix(
    test_data["stroke"],
    test_preds
)

TN, FP, FN, TP = conf_mat.ravel()

sensitivity = TP / (TP + FN)

specificity = TN / (TN + FP)

print("\nAccuracy:", round(accuracy,4))
print("Sensitivity:", round(sensitivity,4))
print("Specificity:", round(specificity,4))

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    test_data["stroke"],
    test_data["test_probs"]
)

auc_value = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.title(
    f"ROC Curve (AUC = {auc_value:.3f})"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

# =========================================================
# BEST THRESHOLD (YOUDEN INDEX)
# =========================================================

youden_index = tpr - fpr

best_index = np.argmax(youden_index)

best_threshold = thresholds[best_index]

print("\nBest Threshold:",
      round(best_threshold, 6))

print("Best Sensitivity:",
      round(tpr[best_index], 4))

print("Best Specificity:",
      round(1 - fpr[best_index], 4))

# =========================================================
# CONFUSION MATRIX HEATMAP
# =========================================================

cm_df = pd.DataFrame(
    conf_mat,
    index=[
        "Actual No Stroke",
        "Actual Stroke"
    ],
    columns=[
        "Predicted No Stroke",
        "Predicted Stroke"
    ]
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix for Stroke Prediction"
)

plt.show()