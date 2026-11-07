# Brain Stroke Dataset: Random Forest Prediction Report

## Overview

This section summarizes and interprets the Random Forest models used to identify variables associated with brain stroke prediction. The models were developed to evaluate which factors were most important for distinguishing stroke cases from non-stroke cases.

The Random Forest models used class weighting to improve sensitivity and address the strong class imbalance in the dataset. Because stroke cases made up only about 5% of the dataset, accuracy alone was not sufficient for evaluating model performance. Sensitivity, specificity, false positives, false negatives, ROC-AUC, and variable importance were also considered.

The analysis was conducted using both **R** and **Python** to compare results across programming environments and support reproducibility. This section includes dataset summaries, model summaries, threshold-based accuracy tests, variable importance results, out-of-bag performance, and interpretation of major findings.

---

# Data Summary

| Dataset | Observations | Stroke Cases | Stroke Rate (%) |
|---|---:|---:|---:|
| Original | 4,891 | 248 | 4.98% |
| Train | 2,491 | 123 | 4.94% |
| Test | 2,490 | 125 | 5.02% |

## Data Split

- The original patient dataset was cleaned and split into a **50% training set** and **50% testing set** using random data partitioning functions in R and Python.
- The splitting algorithm ensured that the stroke rate was approximately similar between the training and testing sets.
- Only the training set was used to construct the Random Forest models.
- The testing set was used to obtain predicted stroke probabilities and evaluate model performance.

---

# Tuned Random Forest Model

## Software Used

- RStudio
- Spyder, Python

---

# Model Summary: R

| Item | Value |
|---|---:|
| Training time | 0.5090 seconds |

## R Model Configuration

| Parameter | Value |
|---|---|
| Model Type | Random Forest Classification |
| Response Variable | Stroke |
| Training Observations | 2,491 |
| Testing Observations | 2,490 |
| Number of Trees, `ntree` | 100 |
| Variables Tried at Each Split, `mtry` | 3 |
| Class Weights | Class 0 = 1, Class 1 = 19 |
| Importance Calculation | Enabled |
| OOB Accuracy | 94.66% |
| OOB Error Rate | 5.34% |

---

# Model Summary: Python

| Item | Value |
|---|---:|
| Training time | 0.2366 seconds |

The Python model training time was approximately half of the R model training time.

## Python Model Configuration

| Parameter | Value |
|---|---|
| Model Type | Random Forest Classification |
| Training Observations | 2,491 |
| Testing Observations | 2,490 |
| Number of Trees, `n_estimators` | 100 |
| Maximum Tree Depth, `max_depth` | 6 |
| Class Weighting | Balanced |
| Random State | 42 |
| Split Criterion | Gini Impurity |
| Features per Split | Auto, √p |
| OOB Accuracy | 95.02% |
| OOB Error | 4.08% |

---

# Accuracy Tests

## Accuracy Test: R

### Threshold = 0.50

| Metric | Value |
|---|---:|
| Test Accuracy | 0.9490 |
| Sensitivity | 0.0160 |
| Specificity | 0.9983 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 2,361 | 123 |
| 1 | 4 | 2 |

At threshold 0.50, the R Random Forest model had very high specificity but very low sensitivity. It correctly identified most non-stroke cases but missed nearly all stroke cases.

---

### Best Threshold = 0.075

| Metric | Value |
|---|---:|
| Test Accuracy | 0.6598 |
| Sensitivity | 0.8080 |
| Specificity | 0.6520 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,542 | 24 |
| 1 | 823 | 101 |

### Key Observations

- The model detected **101 out of 125** actual stroke cases.
- The model missed **24 out of 125** stroke cases.
- The number of false positives increased.
- Overall accuracy decreased because the model predicted more observations as stroke-positive.
- False negatives were substantially reduced.
- The lower threshold improved sensitivity, making the model more suitable for a screening-support context.

---

## Accuracy Test: Python

### Threshold = 0.50

| Metric | Value |
|---|---:|
| Test Accuracy | 0.7884 |
| Sensitivity | 0.6560 |
| Specificity | 0.7953 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,881 | 43 |
| 1 | 484 | 82 |

---

### Best Threshold = 0.0721

| Metric | Value |
|---|---:|
| Test Accuracy | 0.7020 |
| Sensitivity | 0.8080 |
| Specificity | 0.6964 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,647 | 24 |
| 1 | 718 | 101 |

### Key Observations

- The model detected **101 out of 125** actual stroke cases.
- The model missed **24 out of 125** stroke cases.
- False positives increased at the lower threshold.
- Accuracy decreased, but sensitivity improved substantially.

---

# Model Comparison

Random Forest models were implemented in both R and Python using the same training and testing partitions. Both models used class weighting to improve the detection of stroke cases, making them more appropriate for a medical screening-support context than a model focused only on overall accuracy.

Despite minor implementation differences between the software packages, both models achieved comparable performance at their selected lower thresholds. Under similar thresholds, differing by approximately 0.003, both models achieved an identical sensitivity of **80.8%**.

| Metric | R Model | Python Model |
|---|---:|---:|
| Selected Threshold | 0.0750 | 0.0721 |
| Sensitivity | 0.8080 | 0.8080 |
| Specificity | 0.6520 | 0.6964 |
| Test Accuracy | 0.6598 | 0.7020 |
| Stroke Cases Detected | 101 / 125 | 101 / 125 |
| Stroke Cases Missed | 24 / 125 | 24 / 125 |

The Python model produced a specificity of **69.64%**, while the R model produced a specificity of **65.20%**, a difference of about **4.4 percentage points**. Test accuracy also differed by about **4.2 percentage points**.

These results indicate consistent identification of stroke cases across platforms. The main differences likely come from implementation details, such as tree depth restrictions, categorical variable handling, class weighting methods, and default model settings.

---

# Variable Importance

## Variable Importance: R

| Rank | Variable | Mean Decrease Gini | Importance (%) |
|---:|---|---:|---:|
| 1 | age | 108.405 | 47.63% |
| 2 | bmi | 36.848 | 16.19% |
| 3 | avg_glucose_level | 33.792 | 14.85% |
| 4 | work_type | 20.470 | 8.99% |
| 5 | smoking_status | 14.015 | 6.16% |
| 6 | ever_married | 3.813 | 1.68% |
| 7 | Gender | 3.677 | 1.62% |
| 8 | Residence_type | 3.656 | 1.61% |
| 9 | hypertension | 1.587 | 0.70% |
| 10 | heart_disease | 1.334 | 0.59% |
| Total |  | 227.598 | 100.00% |

---

## Variable Importance: Python

| Rank | Variable | Importance | Importance (%) |
|---:|---|---:|---:|
| 1 | age | 0.489237 | 48.92% |
| 2 | avg_glucose_level | 0.127159 | 12.72% |
| 3 | bmi | 0.105354 | 10.54% |
| 4 | hypertension_1 | 0.061344 | 6.13% |
| 5 | ever_married_Yes | 0.056245 | 5.62% |
| 6 | work_type_children | 0.037549 | 3.75% |
| 7 | heart_disease_1 | 0.029821 | 2.98% |
| 8 | smoking_status_never smoked | 0.016268 | 1.63% |
| 9 | work_type_Self-employed | 0.016172 | 1.62% |
| 10 | work_type_Private | 0.014209 | 1.42% |
| 11 | smoking_status_smokes | 0.013645 | 1.36% |
| 12 | Gender_Male | 0.013434 | 1.34% |
| 13 | Residence_type_Urban | 0.010928 | 1.09% |
| 14 | smoking_status_formerly smoked | 0.008636 | 0.86% |

---

# Variable Importance: Key Observations

The variable importance results were derived from class-weighted Random Forest models. These models were modified to address class imbalance and improve stroke detection, rather than focusing only on the majority non-stroke class.

Both the R and Python class-weighted models identified three dominant predictors:

1. **Age**
   - Age had the highest variable importance in both models.
   - It accounted for approximately **47% to 49%** of the total importance.
   - This suggests that age was one of the strongest model-associated variables for distinguishing stroke from non-stroke patients in the dataset.

2. **Average Glucose Level**
   - Average glucose level consistently appeared among the top three variables.
   - It accounted for approximately **12% to 15%** of the total importance.

3. **BMI**
   - BMI was also one of the dominant predictors.
   - In the R model, BMI ranked second.
   - In the Python model, BMI ranked third.

Additional observations:

- Hypertension had a moderate contribution in the Python Random Forest, but its ranking was lower in the R Random Forest.
- Smoking status and marital status were more prominent in the Random Forest models than they were in some earlier models, such as the Decision Tree and Logistic Regression models.
- Variable importance should be interpreted as **model-associated importance only**. It does not prove causation or statistical significance.

---

# Train vs. Test Accuracy

| Software | Train Accuracy | Test Accuracy |
|---|---:|---:|
| R | 1.0000 | 0.6598 |
| Python | 0.8222 | 0.7020 |

## Key Observations

- The R Random Forest achieved perfect training accuracy, but test accuracy dropped from **100.00%** to **65.98%**.
- This suggests significant overfitting, meaning the R model may have captured patterns specific to the training data rather than generalizable patterns.
- The Python Random Forest showed a smaller gap between training and testing accuracy.
- Although the Python model was less accurate on the training set, it achieved higher test accuracy.
- This suggests that the Python model had a better balance between learning from the training data and maintaining predictive performance on new observations.

---

# Out-of-Bag Score

| Software | OOB Accuracy | OOB Error |
|---|---:|---:|
| R | 94.66% | 5.34% |
| Python | 95.02% | 4.08% |

## Key Observations

- Both models achieved very high OOB accuracy above 94%.
- This indicates that both models classified the majority class, non-stroke cases, effectively.
- The Python model produced slightly better OOB results, with higher OOB accuracy and lower OOB error.
- However, OOB accuracy is heavily influenced by class imbalance because approximately 95% of the dataset consists of non-stroke cases.
- Therefore, OOB accuracy should be interpreted alongside sensitivity, specificity, false positives, false negatives, and precision.

---

# Interpretation Summary

The Random Forest models identified **age**, **average glucose level**, and **BMI** as the most important model-associated predictors of stroke classification. Age was the strongest predictor across both R and Python models, accounting for nearly half of the total variable importance.

The results also showed that threshold selection strongly affected model performance. At the default threshold of 0.50, some models had high accuracy and specificity but poor sensitivity, meaning many stroke cases were missed. Lowering the threshold substantially improved sensitivity and allowed the model to detect more actual stroke cases, but this came at the cost of more false positives and lower specificity.

For stroke prediction, sensitivity is especially important because false negatives represent missed stroke cases. Therefore, lower thresholds may be more appropriate for a screening-support model. However, the high number of false positives and low precision mean that the model is not clinically ready and should not be used as a standalone diagnostic tool.

---

# Conclusion

The Random Forest models provided useful information about variables associated with stroke prediction. Across both R and Python, **age**, **average glucose level**, and **BMI** were consistently identified as the most important model-associated predictors.

The class-weighted models improved sensitivity and reduced false negatives, making them more appropriate for a stroke-screening support context than models optimized only for accuracy. However, this improvement came with increased false positives and reduced specificity.

Overall, the Random Forest analysis supports the conclusion that age, glucose level, and BMI are important variables in predicting stroke probability within this dataset. The results should be interpreted as model-associated relationships only, not as evidence of causation or statistical significance. Further validation, calibration, and clinical review would be required before any real-world medical use.
