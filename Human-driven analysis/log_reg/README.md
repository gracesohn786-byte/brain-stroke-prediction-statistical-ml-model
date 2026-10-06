# Brain Stroke Dataset: Logistic Regression Results Report

## Overview

This section summarizes and interprets the results of two logistic regression models used to examine predictors associated with brain stroke occurrence. The outcome variable, `stroke`, is binary:

- `0` = No Stroke
- `1` = Stroke

Two logistic regression models were constructed:

1. **Main-effects logistic regression model**
   - Includes the full set of predictors as explanatory variables.
   - Does not include interaction terms.

2. **Interaction logistic regression model**
   - Includes selected predictors and interaction terms.
   - Used to examine whether combinations of predictors may improve model performance.

The interaction model was selected for further interpretation because it showed stronger predictive performance. Each model section includes model summaries, coefficient tables, odds ratios, model accuracy results, threshold-based evaluation, and ROC curve interpretation.

The models were constructed using both **R** and **Python** to compare results across programming environments and support reproducibility. The analysis concludes with an interpretation of major statistical findings, comparison of model performance, and an overall conclusion regarding predictors most strongly associated with stroke probability.

---

## Data Summary

| Dataset | Observations | Stroke Cases | Stroke Rate (%) |
|---|---:|---:|---:|
| Original | 4,891 | 248 | 4.98% |
| Train | 2,491 | 123 | 4.94% |
| Test | 2,490 | 125 | 5.02% |

### Data Split

- The original patient dataset was split into a **50% training set** and **50% testing set** using random data partitioning functions in R and Python.
- The splitting algorithm ensured that the stroke rate was approximately similar between the training and testing sets.
- Only the training set was used to construct the logistic regression models.
- The testing set was used to obtain predicted stroke probabilities and evaluate overall model performance.

---

## Predictors

The original dataset includes numerical and categorical predictors.

### Numerical Predictors

- `age`
- `avg_glucose_level`
- `bmi`

### Categorical Predictors

- `Gender`: 2 levels
- `hypertension`: 2 levels
- `heart_disease`: 2 levels
- `ever_married`: 2 levels
- `work_type`: 4 levels
- `Residence_type`: 2 levels
- `smoking_status`: 4 levels

> Note: The variable `stroke` is the target variable and is not used as a predictor.

---

# Model 1: Main-Effects Logistic Regression

## Model Summary

| Item | Value |
|---|---:|
| Predictors | All variables |
| Dependent variable | Stroke |
| Interaction terms | None |
| Observations | 2,491 |
| Degrees of Freedom, Model | 11 |
| Degrees of Freedom, Residual | 2,479 |
| R training time | 0.0596 seconds |
| Python training time | 0.0295 seconds |

---

## Coefficients Table: R

| Predictor | Estimate | Std. Error | z value | p-value |
|---|---:|---:|---:|---:|
| Intercept | -8.660846 | 0.856115 | -10.116 | < 2e-16 |
| Gender: Male | 0.024554 | 0.203477 | 0.121 | 0.90395 |
| Age | 0.073622 | 0.008056 | 9.139 | < 2e-16 |
| Hypertension = 1 | 0.614048 | 0.223214 | 2.751 | 0.00594 |
| Heart Disease = 1 | 0.329359 | 0.280718 | 1.173 | 0.24069 |
| Ever Married = Yes | -0.142399 | 0.328093 | -0.434 | 0.66427 |
| Residence Type = Urban | -0.029754 | 0.198598 | -0.150 | 0.88091 |
| Average Glucose Level | 0.004125 | 0.001707 | 2.416 | 0.01568 |
| BMI | 0.030499 | 0.017214 | 1.772 | 0.07643 |
| Smoking Status = Never Smoked | -0.276846 | 0.262624 | -1.054 | 0.29181 |
| Smoking Status = Smokes | 0.509917 | 0.301809 | 1.690 | 0.09112 |
| Smoking Status = Unknown | 0.238734 | 0.290951 | 0.821 | 0.41191 |

### R Model Fit

| Fit Statistic | Value |
|---|---:|
| Null deviance | 979.85 on 2,490 df |
| Residual deviance | 760.58 on 2,479 df |
| Fisher scoring iterations | 8 |

---

## Coefficients Table: Python

| Variable | Coefficient (β) | Std. Error | z-value | p-value | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|---:|---:|
| Intercept | -8.4221 | 0.839 | -10.038 | <0.001 | -10.067 | -6.778 |
| Gender: Male

# Logistic Regression Model Interpretation and Interaction Model Results

## Main-Effects Model: Key Observations

The main-effects logistic regression model identified **age**, **hypertension**, and **average glucose level** as statistically significant positive predictors associated with stroke probability.

### Age

Age was found to be a highly significant predictor of stroke.

| Measure | Value |
|---|---:|
| Odds Ratio | 1.0764 |
| Approximate percent increase in odds | 7.64% per additional year |
| p-value | < 2e-16 |
| Confidence interval | Entirely above 1 |

The odds ratio for age was **1.0764**, which indicates that for each additional year of age, the model-estimated odds of stroke increased by approximately:

```text
(1.0764 - 1) × 100 = 7.64%
```

This suggests that older age was strongly associated with higher predicted stroke odds in the model. The p-value was extremely small, and the confidence interval was entirely above 1 in both R and Python, supporting the conclusion that age was statistically significant in the main-effects model.

### Hypertension

Hypertension was also statistically significant.

| Measure | Value |
|---|---:|
| Odds Ratio | 1.8740 |
| Approximate percent increase in odds | 87.40% |
| Confidence interval | 1.20 to 2.89 |

The odds ratio for hypertension was approximately **1.8740**, meaning that individuals with hypertension had about **87% higher model-estimated odds of stroke** compared with individuals without hypertension, holding other variables constant:

```text
(1.8740 - 1) × 100 = 87.40%
```

The confidence interval for hypertension was entirely above 1, supporting the conclusion that hypertension was significantly associated with increased stroke odds in the model.

### Average Glucose Level

Average glucose level was also statistically significant, although the effect size was small per one-unit increase.

| Measure | Value |
|---|---:|
| Odds Ratio | 1.0041 |
| p-value | 0.017 |

The odds ratio for average glucose level was **1.0041**, indicating a small positive association between glucose level and model-estimated stroke odds. Although the effect per one-unit increase was small, higher average glucose levels were associated with higher predicted stroke probability in the model.

### Summary of Main-Effects Model

Overall, the main-effects logistic regression model showed that **age**, **hypertension**, and **average glucose level** had statistically significant positive associations with stroke probability. Age and hypertension showed especially notable effects. These findings should be interpreted as statistical associations within the model, not as evidence of causation.

---

# Model 2: Interaction Logistic Regression Model

## Model Summary

| Item | Value |
|---|---:|
| Predictors | Gender, age, BMI, heart disease, hypertension, average glucose level |
| Dependent variable | Stroke |
| Interaction terms | Age × BMI, BMI × Average Glucose Level |
| Observations | 2,491 |
| Degrees of Freedom, Model | 8 |
| Degrees of Freedom, Residual | 2,482 |
| R training time | 0.2322 seconds |
| Python training time | 0.0168 seconds |

---

## Coefficients Table: R

| Variable | Estimate | Std. Error | z value | p-value |
|---|---:|---:|---:|---:|
| Intercept | -9.4912 | 2.5608 | -3.7060 | 0.0002 |
| Gender: Male | 0.1148 | 0.2007 | 0.5720 | 0.5674 |
| Age | 0.1237 | 0.0346 | 3.5720 | 0.0004 |
| BMI | 0.0628 | 0.0803 | 0.7820 | 0.4343 |
| Heart Disease = 1 | 0.4254 | 0.2778 | 1.5310 | 0.1257 |
| Hypertension = 1 | 0.5838 | 0.2211 | 2.6410 | 0.0083 |
| Average Glucose Level | -0.0149 | 0.0085 | -1.7520 | 0.0798 |
| Age × BMI | -0.0018 | 0.0011 | -1.6210 | 0.1050 |
| BMI × Average Glucose Level | 0.0006 | 0.0003 | 2.2940 | 0.0218 |

### R Model Fit

| Fit Statistic | Value |
|---|---:|
| Null deviance | 979.85 on 2,490 df |
| Residual deviance | 761.28 on 2,482 df |
| Fisher scoring iterations | 8 |

---

## Coefficients Table: Python

| Variable | Coefficient | Std. Error | z-value | p-value | Lower 95% CI | Upper 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Intercept | -9.4912 | 2.561 | -3.706 | <0.001 | -14.510 | -4.472 |
| Gender: Male | 0.1148 | 0.201 | 0.572 | 0.567 | -0.279 | 0.508 |
| Heart Disease | 0.4254 | 0.278 | 1.531 | 0.126 | -0.119 | 0.970 |
| Hypertension | 0.5838 | 0.221 | 2.641 | 0.008 | 0.150 | 1.017 |
| Age | 0.1237 | 0.035 | 3.572 | <0.001 | 0.056 | 0.192 |
| BMI | 0.0628 | 0.080 | 0.782 | 0.434 | -0.095 | 0.220 |
| Age × BMI | -0.0018 | 0.001 | -1.621 | 0.105 | -0.004 | 0.000 |
| Average Glucose Level | -0.0149 | 0.009 | -1.752 | 0.080 | -0.032 | 0.002 |
| BMI × Average Glucose Level | 0.0006 | 0.000 | 2.294 | 0.022 | 0.000088 | 0.0010 |

---

## Odds Ratios

| Variable | Odds Ratio: R | Odds Ratio: Python |
|---|---:|---:|
| Intercept | 7.5517e-05 | 0.000076 |
| Gender: Male | 1.1216 | 1.1216 |
| Age | 1.1317 | 1.1317 |
| BMI | 1.0648 | 1.0648 |
| Heart Disease = 1 | 1.5302 | 1.5302 |
| Hypertension = 1 | 1.7929 | 1.7929 |
| Average Glucose Level | 0.9852 | 0.9852 |
| Age × BMI | 0.9982 | 0.9982 |
| BMI × Average Glucose Level | 1.0006 | 1.0006 |

---

## Odds Ratio Confidence Intervals

| Variable | R 2.5% CI | R 97.5% CI | Python Lower 95% CI | Python Upper 95% CI |
|---|---:|---:|---:|---:|
| Intercept | 4.6754e-07 | 0.0101 | 0.0000005 | 0.0114 |
| Gender: Male | 0.7548 | 1.6603 | 0.7569 | 1.6621 |
| Age | 1.0588 | 1.2118 | 1.0574 | 1.2111 |
| BMI | 0.9075 | 1.2413 | 0.9098 | 1.2462 |
| Heart Disease = 1 | 0.8716 | 2.5997 | 0.8878 | 2.6375 |
| Hypertension = 1 | 1.1536 | 2.7491 | 1.1624 | 2.7654 |
| Average Glucose Level | 0.9688 | 1.0017 | 0.9689 | 1.0018 |
| Age × BMI | 0.9961 | 1.0004 | 0.9960 | 1.0004 |
| BMI × Average Glucose Level | 1.0001 | 1.0011 | 1.0001 | 1.0011 |

---

# Accuracy Test: R

## Threshold = 0.05

| Metric | Value |
|---|---:|
| Sensitivity | 0.8080 |
| Specificity | 0.7328 |
| Accuracy | 0.7365 |

At this threshold, the model correctly identified about **81% of actual stroke cases**. This threshold reduced false negatives but increased false positives.

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,733 | 24 |
| 1 | 632 | 101 |

---

## Threshold = 0.10

| Metric | Value |
|---|---:|
| Sensitivity | 0.6320 |
| Specificity | 0.8457 |
| Accuracy | 0.8349 |

At this threshold, the model detected a more moderate proportion of stroke cases while reducing false positives.

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 2,000 | 46 |
| 1 | 365 | 79 |

---

## Threshold = 0.20

| Metric | Value |
|---|---:|
| Sensitivity | 0.2640 |
| Specificity | 0.9476 |
| Accuracy | 0.9133 |

This threshold produced the highest accuracy and was good at identifying non-stroke individuals. However, it detected only **26.4% of stroke cases**, making it less useful for a stroke-screening context.

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 2,241 | 92 |
| 1 | 124 | 33 |

---

## Best Threshold from ROC Curve = 0.04872067

| Metric | Value |
|---|---:|
| Sensitivity | 0.8240 |
| Specificity | 0.7294 |
| Accuracy | 0.7341 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,725 | 22 |
| 1 | 640 | 103 |

---

# Accuracy Test: Python

## Threshold = 0.05

| Metric | Value |
|---|---:|
| Sensitivity | 0.8080 |
| Specificity | 0.7328 |
| Accuracy | 0.7365 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,733 | 24 |
| 1 | 632 | 101 |

---

## Best Threshold from ROC Curve = 0.048734

| Metric | Value |
|---|---:|
| Sensitivity | 0.8160 |
| Specificity | 0.7294 |
| Accuracy | 0.7337 |

### Confusion Matrix

| Predicted / Actual | 0 | 1 |
|---|---:|---:|
| 0 | 1,725 | 23 |
| 1 | 640 | 102 |

---

# Interaction Model: Key Observations

The interaction logistic regression model included selected main effects and two interaction terms: **Age × BMI** and **BMI × Average Glucose Level**. This model was used to evaluate whether combinations of predictors were associated with stroke probability.

## Age

Age remained statistically significant in the interaction model.

| Measure | Value |
|---|---:|
| Coefficient | 0.1237 |
| Odds Ratio | 1.1317 |
| p-value | <0.001 |
| 95% CI | Entirely above 1 |

The odds ratio for age was **1.1317**, which suggests that, when the interacting terms are included, age was positively associated with model-estimated stroke odds. However, because age is part of an interaction with BMI, the effect of age should not be interpreted independently in the same way as in a model without interactions.

## Hypertension

Hypertension was also statistically significant in the interaction model.

| Measure | Value |
|---|---:|
| Coefficient | 0.5838 |
| Odds Ratio | 1.7929 |
| p-value | 0.008 |
| 95% CI | Entirely above 1 |

The odds ratio for hypertension was **1.7929**, indicating that individuals with hypertension had higher model-estimated odds of stroke than individuals without hypertension, holding other variables constant.

## BMI × Average Glucose Level Interaction

The interaction between BMI and average glucose level was statistically significant.

| Measure | Value |
|---|---:|
| Coefficient | 0.0006 |
| Odds Ratio | 1.0006 |
| p-value | 0.0218 |
| 95% CI | Entirely above 1 |

The statistically significant BMI × average glucose level interaction suggests that the relationship between average glucose level and stroke probability may depend partly on BMI. Although the odds ratio is close to 1 because the interaction coefficient is small, the confidence interval was above 1, indicating a statistically significant positive interaction in the model.

## Non-significant Predictors

The following variables were not statistically significant at the conventional 0.05 level in the interaction model:

- Gender
- BMI main effect
- Heart disease
- Average glucose level main effect
- Age × BMI interaction

This does not mean these variables have no relationship with stroke risk. It means that, in this model and dataset, their estimated effects were not statistically significant after accounting for the other included predictors and interaction terms.

---

# Interaction Model Interpretation Summary

The interaction logistic regression model showed that **age**, **hypertension**, and the **BMI × average glucose level interaction** were statistically significant predictors associated with stroke probability.

The model suggests that stroke probability increased with age and was higher among individuals with hypertension. The significant BMI × average glucose level interaction suggests that the relationship between glucose level and stroke probability may vary depending on BMI.

Threshold selection had a major effect on model performance. At lower thresholds such as **0.05** and the ROC-based threshold of approximately **0.049**, the model achieved higher sensitivity and identified more stroke cases. However, this came with more false positives and lower specificity. At the higher threshold of **0.20**, the model had higher accuracy and specificity but missed many stroke cases.

Because stroke cases were rare in the dataset, accuracy alone was not sufficient for evaluating the model. Sensitivity, specificity, ROC-AUC, PR-AUC, and the confusion matrix provide more meaningful insight into model performance.

The interaction model results should be interpreted as **statistical and model-associated relationships**, not as evidence of causation. The model is not clinically ready and should not be used as a diagnostic tool without further validation.

---

# Conclusion

The interaction logistic regression model provided an interpretable statistical approach for examining factors associated with stroke probability. Compared with the main-effects model, it allowed the analysis to evaluate whether combined effects, such as BMI interacting with average glucose level, were related to stroke probability.

The most important statistically significant findings from the interaction model were:

- **Age** was positively associated with stroke probability.
- **Hypertension** was positively associated with stroke probability.
- **BMI × Average Glucose Level** was a statistically significant positive interaction.

In a stroke-screening context, lower classification thresholds were more useful because they improved sensitivity and identified more actual stroke cases. However, this came with a tradeoff of increased false positives. Therefore, while the interaction logistic regression model may be useful as an exploratory statistical model, it should not be considered clinically ready.

# Interaction Model Key Observations and Model Comparison

## Interaction Model: Key Observations

The interaction logistic regression model included selected main-effect predictors and two interaction terms:

- `age × bmi`
- `bmi × avg_glucose_level`

This model was used to examine whether combinations of predictors improved the explanation of stroke probability compared with the main-effects model.

### Age

Age was statistically significant in the interaction model.

| Measure | Value |
|---|---:|
| p-value | 0.0035 |
| Interpretation | Age was significantly associated with stroke probability |

This result suggests that age remained an important predictor of stroke probability even after including interaction terms. Older age was associated with higher model-estimated stroke probability.

### Hypertension

Hypertension was also statistically significant.

| Measure | Value |
|---|---:|
| p-value | 0.0083 |
| Interpretation | Hypertension was significantly associated with stroke probability |

This indicates that individuals with hypertension had higher model-estimated stroke probability compared with individuals without hypertension, holding other predictors constant.

### BMI × Average Glucose Level Interaction

The interaction between BMI and average glucose level was statistically significant.

| Measure | Value |
|---|---:|
| p-value | 0.0218 |
| Interpretation | The combined effect of BMI and average glucose level was significantly associated with stroke probability |

Although BMI was not statistically significant as an independent predictor in this model, the significant `bmi × avg_glucose_level` interaction suggests that BMI may become more influential when average glucose levels increase. In other words, the relationship between glucose level and stroke probability may depend partly on BMI.

This interaction should be interpreted carefully. It does not mean that BMI alone is a significant predictor, but rather that the combination of BMI and glucose level contributed meaningfully to the model.

### Age × BMI Interaction

The interaction between age and BMI was not statistically significant.

| Measure | Value |
|---|---:|
| p-value | 0.1050 |
| Interpretation | The age-BMI interaction was not statistically significant |

This suggests that the effect of age on stroke probability did not significantly change across BMI levels in this model. Age remained important, but there was not strong statistical evidence that BMI modified the relationship between age and stroke probability.

---

# Model Comparison

## AIC Comparison

| Model | AIC |
|---|---:|
| Main-effects model | 784.58 |
| Interaction model | 779.28 |

The interaction model had a lower AIC score than the main-effects model. Since lower AIC values indicate better relative model fit, this suggests that the interaction model demonstrated a modest improvement in fit compared with the main-effects model.

This improvement suggests that combined effects among predictors, especially the interaction between BMI and average glucose level, may capture meaningful relationships associated with stroke probability.

However, the improvement was modest, so the interaction model should be interpreted as a slightly better-fitting model rather than a dramatically superior model.

---

# Similarities Across Models

Both the main-effects model and the interaction model showed similar patterns in threshold-based performance.

## Threshold Effects

| Threshold Choice | Effect |
|---|---|
| Lower threshold | More stroke-positive predictions |
| Lower threshold | Higher sensitivity |
| Lower threshold | More false positives |
| Higher threshold | Fewer stroke-positive predictions |
| Higher threshold | Higher specificity |
| Higher threshold | More false negatives |
| Higher threshold | Lower sensitivity |

Lower thresholds made the models more likely to classify individuals as stroke-positive. This improved sensitivity because more actual stroke cases were detected, but it also increased the number of false positives.

Higher thresholds made the models more conservative. This improved specificity and accuracy, but it caused the models to miss more actual stroke cases.

---

# Accuracy, Sensitivity, and Class Imbalance

The dataset was highly imbalanced, with stroke cases making up only about **5%** of the observations. Because of this imbalance, accuracy alone was not an adequate measure of model performance.

A model could achieve high accuracy by predicting most individuals as stroke-negative, but this would result in poor stroke detection and a high number of false negatives.

In the context of stroke prediction, false negatives are especially important because they represent actual stroke cases that the model failed to identify.

---

# Medical Screening Interpretation

Given the medical context of stroke prediction, thresholds that emphasize sensitivity may be more appropriate when the goal is screening or identifying potentially high-risk individuals.

A sensitivity-focused threshold is useful because it detects more actual stroke cases. However, this comes with the tradeoff of more false positives, meaning more individuals without stroke are incorrectly flagged as stroke-positive.

Therefore, the model may be useful as an exploratory screening-support tool, but it is not clinically ready and should not be used as a diagnostic model without further validation.

---

# Summary

The interaction model showed that:

- Age was statistically significant.
- Hypertension was statistically significant.
- The BMI × average glucose level interaction was statistically significant.
- The age × BMI interaction was not statistically significant.
- The interaction model had a lower AIC than the main-effects model, suggesting modest improvement in fit.
- Threshold choice strongly affected sensitivity, specificity, false positives, and false negatives.
- Because stroke cases were rare, accuracy alone was misleading.
- In a stroke-screening context, sensitivity is especially important, but the model is not clinically ready.

Overall, the interaction model provided a slightly better fit than the main-effects model and suggested that the combined effect of
