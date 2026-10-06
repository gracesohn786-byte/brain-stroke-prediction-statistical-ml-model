# Brain Stroke Dataset: Decision Tree Prediction Report

## I. Overview

This section summarizes and interprets the Decision Tree models used to identify variables associated with brain stroke prediction. Three decision tree models were evaluated:

1. **Original Decision Tree**
2. **Balanced Decision Tree**
3. **Tuned Decision Tree**

The balanced and tuned models were modified from the original model to improve stroke detection and address the strong class imbalance in the dataset. The final tuned decision tree model was selected because it provided better interpretability and improved prediction performance compared with the original tree.

The analysis includes model summaries, variable importance, threshold-based accuracy tests, ROC curve interpretation, and decision tree plots from both **R** and **Python**. The section concludes with an interpretation of key findings and an overall summary of the predictors most strongly associated with stroke classification.

---

## II. Data Summary

| Dataset | Observations | Stroke Cases | Stroke Rate (%) |
|---|---:|---:|---:|
| Original | 4,891 | 248 | 4.98% |
| Train | 2,491 | 123 | 4.94% |
| Test | 2,490 | 125 | 5.02% |

### Data Split

- The original patient dataset was split into a **50% training set** and **50% testing set** using random data partitioning functions in R and Python.
- The splitting algorithm ensured that the stroke rate was approximately similar between the training and testing sets.
- Only the training set was used to construct the decision tree models.
- The testing set was used to obtain predicted stroke probabilities and evaluate model performance.

---

# III. Decision Tree Models

---

# 1. Original Decision Tree Model

## Software Used

- RStudio
- Spyder, Python

## Training Time

| Software | Training Time |
|---|---:|
| R / Python workflow | 0.0566 seconds |

---

## Model Summary

| Component | Value |
|---|---|
| Formula | `stroke ~ Gender + age + hypertension + heart_disease + ever_married + work_type + Residence_type + avg_glucose_level + bmi + smoking_status` |
| Method | Classification tree |
| Training observations | 2,491 |
| Complexity parameter, CP | 0.008130081 |
| Number of splits | 0 |
| Relative error | 1.0000 |
| Cross-validation error | 0 |
| Cross-validation standard deviation | 0 |
| Predicted class | 0 |
| Expected loss | 0.04937776 |
| Class counts | 2,368 non-stroke, 123 stroke |
| Class probabilities | 0.951 non-stroke, 0.049 stroke |
| Variable importance | NULL |

---

## Key Observations

- The original decision tree did **not split at all**.
- This indicates that the dataset was highly imbalanced:
  - Approximately **95%** non-stroke cases
  - Approximately **5%** stroke cases
- The model likely minimized classification error by predicting the majority class, **No Stroke**, for almost every patient.
- This is consistent with the class probabilities:
  - 0.951 probability for non-stroke
  - 0.049 probability for stroke
- Because no splits occurred, no predictors were used in the tree.
- As a result, variable importance was `NULL`.
- The class imbalance was severe enough that a standard decision tree could not effectively detect stroke patterns.

---

## Accuracy Test: Original Tree

### Threshold = 0.50

| Metric | Value |
|---|---:|
| Test Accuracy | 0.9470 |
| Sensitivity | 0.0161 |
| Specificity | 0.9958 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 2,357 | 10 |
| 1 | 122 | 2 |

At the default threshold, the original decision tree achieved high accuracy and specificity but extremely low sensitivity. This means the model classified most non-stroke cases correctly but missed nearly all stroke cases.

---

# 2. Balanced Decision Tree Model

## Software Used

- RStudio

## Model Feature

The balanced tree model used class balancing to address the stroke/non-stroke imbalance.

- The balanced training dataset included:
  - 123 non-stroke cases
  - 123 stroke cases
- This created a 50% stroke and 50% non-stroke training distribution.
- The goal was to force the tree to better identify patterns among stroke cases.

---

## Model Summary

| Component | Result |
|---|---|
| Model type | Classification tree, `method = "class"` |
| Training observations | 246 |
| Class distribution | 123 non-stroke, 123 stroke |
| Tree splits | 3 splits |
| Main splitting variable | Age |
| Most important variables | Age, ever_married, work_type, BMI, hypertension |
| Root node accuracy | 50% / 50% |
| First split | Age < 48.5 |
| Best stroke prediction region | Older individuals, ≥48.5 |
| Highest stroke probability node | Node 13 |
| Lowest stroke probability node | Node 2 |

---

## Variable Importance Table

| Variable | Importance Score |
|---|---:|
| Age | 49.96 |
| ever_married | 8.51 |
| work_type | 7.94 |
| BMI | 6.47 |
| hypertension | 3.22 |

---

## Complexity Parameter Table

| CP | Number of Splits | Relative Error | Cross-Validation Error | xstd |
|---:|---:|---:|---:|---:|
| 0.56910569 | 0 | 1.0000000 | 1.1788618 | 0.06272953 |
| 0.03252033 | 1 | 0.4308943 | 0.4634146 | 0.05380166 |
| 0.01000000 | 3 | 0.3658537 | 0.4471545 | 0.05312823 |

---

## Key Observations

- **Age was the strongest predictor** in the balanced decision tree.
  - Importance score: approximately 50
  - Age was used at the root node, meaning it produced the greatest reduction in classification impurity.
  - Patients younger than approximately 48 years were mostly classified as non-stroke.
- This supports earlier logistic regression findings where age showed the strongest statistical significance.
- **Hypertension emerged as an important predictor** in later branches.
  - It appeared in later nodes among older patients.
  - Older patients with hypertension, higher BMI, and higher glucose values showed higher stroke probability.
- This suggests that hypertension may interact with age in identifying stroke risk patterns, which is consistent with the interaction logistic regression model.
- **BMI contributed moderately** by refining predictions.
- **Work type** and **marital status** were also identified as meaningful classification variables.
- Overall, the balanced tree produced more clinically useful stroke detection than the original tree.

---

## Accuracy Test: Balanced Tree

### Best Threshold = 0.531295

| Metric | Value |
|---|---:|
| Test Accuracy | 0.6671 |
| Sensitivity | 0.8560 |
| Specificity | 0.6571 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,554 | 18 |
| 1 | 811 | 107 |

The balanced tree detected **107 out of 125** stroke cases, producing strong sensitivity. However, this improvement came with more false positives and lower specificity.

---

# 3. Tuned Decision Tree Model

## Software Used

- RStudio
- Spyder, Python

The tuned decision tree model adjusted tree complexity settings such as minimum split size, maximum depth, and complexity pruning parameters. The goal was to improve model performance while maintaining interpretability.

---

# Model Summary: R

## Training Time

| Software | Training Time |
|---|---:|
| R | 0.0372 seconds |

## R Model Configuration

| Setting | Value |
|---|---:|
| Method | Classification Tree, `rpart` |
| Dataset size | 2,491 observations |
| minsplit | 10 |
| cp | 0.0001 |
| maxdepth | 5 |

---

## R Complexity Parameter Table

| CP | Number of Splits | Relative Error | Cross-Validation Error | xstd |
|---:|---:|---:|---:|---:|
| 0.010163 | 0 | 1.0000 | 1.0000 | 0.0879 |
| 0.008130 | 4 | 0.9593 | 1.0488 | 0.0899 |
| 0.004065 | 6 | 0.9431 | 1.1382 | 0.0935 |
| 0.000100 | 10 | 0.9268 | 1.2358 | 0.0971 |

The cross-validation error increased as the tree became more complex, suggesting possible overfitting.

---

# Model Summary: Python

## Training Time

| Software | Training Time |
|---|---:|
| Python | 0.0148 seconds |

## Python Model Configuration

| Metric | Value |
|---|---:|
| Tree depth | 5 |
| Number of nodes | 39 |
| Number of leaves | 20 |
| Max depth parameter | 5 |
| Min samples split | 10 |
| Min samples leaf | 5 |
| CCP alpha | 0.0001 |

---

## Best-Pruned Tree Based on Minimum Cross-Validation Error

| Metric | Value |
|---|---:|
| CCP alpha | 0.002153 |
| Leaves | 2 |
| Tree depth | 1 |
| Train error | 0.0494 |
| CV error | 0.0494 |
| CV standard deviation | 0.0018 |

---

# Variable Importance

## Variable Importance: R

| Rank | Variable | Importance Score |
|---:|---|---:|
| 1 | age | 50 |
| 2 | avg_glucose_level | 16 |
| 3 | BMI | 15 |
| 4 | ever_married | 7 |
| 5 | work_type | 6 |
| 6 | hypertension | 5 |
| 7 | smoking_status | 1 |
| 8 | Residence_type | 1 |

---

## Variable Importance: Python

| Rank | Variable | Importance | Importance (%) |
|---:|---|---:|---:|
| 1 | age | 0.452082 | 45.21% |
| 2 | avg_glucose_level | 0.176995 | 17.70% |
| 3 | BMI | 0.176192 | 17.62% |
| 4 | hypertension_1 | 0.058778 | 5.88% |
| 5 | work_type_Private | 0.039236 | 3.92% |
| 6 | heart_disease_1 | 0.037456 | 3.75% |
| 7 | ever_married_Yes | 0.030903 | 3.09% |
| 8 | smoking_status_never smoked | 0.028358 | 2.84% |
| 9 | Gender_Male | 0.000000 | 0.00% |
| 10 | work_type_Self-employed | 0.000000 | 0.00% |
| 11 | smoking_status_formerly smoked | 0.000000 | 0.00% |
| 12 | Residence_type_Urban | 0.000000 | 0.00% |
| 13 | work_type_children | 0.000000 | 0.00% |
| 14 | smoking_status_smokes | 0.000000 | 0.00% |

---

# Accuracy Test: Tuned Tree

## Accuracy Test: R

### Best Threshold = 0.04865

| Metric | Value |
|---|---:|
| Test Accuracy | 0.6056 |
| Sensitivity | 0.8960 |
| Specificity | 0.5903 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,396 | 13 |
| 1 | 969 | 112 |

### Key Interpretation

- The model correctly identified most stroke cases.
- Sensitivity was approximately **89.6%**.
- This means the model detected **112 out of 125** stroke cases.
- However, the model produced many false positives, which lowered specificity.

---

## Accuracy Test: Python

### Best Threshold = 0.0512

| Metric | Value |
|---|---:|
| Test Accuracy | 0.6916 |
| Sensitivity | 0.8240 |
| Specificity | 0.6846 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,619 | 22 |
| 1 | 746 | 103 |

### Key Interpretation

- The Python tuned tree detected **103 out of 125** stroke cases.
- Sensitivity was approximately **82.4%**.
- Specificity was higher than the R tuned tree, meaning the Python model produced fewer false positives.
- The model still sacrificed specificity to improve sensitivity.

---

# Tuned Tree: Key Observations

- **Age was the strongest predictor** across both R and Python.
  - It was the top split and had the highest importance score.
- **Average glucose level** and **BMI** were also major contributors.
- **Hypertension** contributed meaningfully to classification.
- These findings are consistent with earlier logistic regression results, where age, hypertension, and glucose level were important predictors.
- Cross-validation error increased as the R tree became more complex, suggesting possible overfitting.
- Smoking status and residence type had weak influence based on their low importance scores.
- The tuned tree improved sensitivity compared with the original decision tree, making it more useful for stroke-screening support.
- However, the tuned tree produced many false positives and is not clinically ready.

---

# Model Comparison Summary

| Model | Sensitivity | Specificity | Main Strength | Main Limitation |
|---|---:|---:|---|---|
| Original Tree | 0.0161 | 0.9958 | Very high specificity | Missed nearly all stroke cases |
| Balanced Tree | 0.8560 | 0.6571 | Strong stroke detection | Many false positives |
| Tuned Tree: R | 0.8960 | 0.5903 | Highest sensitivity | Lowest specificity |
| Tuned Tree: Python | 0.8240 | 0.6846 | Strong sensitivity with better specificity than R tuned tree | Still many false positives |

---

# Final Interpretation

The original decision tree was not useful for stroke prediction because it did not split and mostly predicted the majority class, non-stroke. This produced high overall accuracy but very poor sensitivity.

The balanced decision tree improved stroke detection by training on a balanced subset of stroke and non-stroke cases. This increased sensitivity substantially but also increased false positives.

The tuned decision tree provided the best balance of interpretability and prediction performance. Across both R and Python, **age** was consistently identified as the strongest model-associated predictor. **Average glucose level**, **BMI**, and **hypertension** were also important contributors.

Because the dataset was highly imbalanced, accuracy alone was misleading. A model could achieve high accuracy by predicting most cases as non-stroke, but this would fail to identify actual stroke cases. Sensitivity was therefore especially important for evaluating the models in a stroke-screening context.

Overall, the tuned decision tree was more useful as a screening-support model than the original tree because it detected many more stroke cases. However, it produced many false positives and should not be considered clinically ready. The results should be interpreted as model-associated patterns only, not as evidence of causation or statistical significance.
