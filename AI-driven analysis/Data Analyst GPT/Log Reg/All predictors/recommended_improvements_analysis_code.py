
"""
Recommended next-improvement analyses for AI-assisted stroke prediction project.

Inputs:
- train_dataset.csv
- test_dataset.csv
- logistic_regression_stroke_model.pkl

Outputs:
- recommended_improvements_summary.csv
- calibration_curve_data.csv
- test_calibration_curve.png
- statsmodels_logistic_or_ci_pvalues.csv
- multicollinearity_vif.csv
- model_comparison_test_results.csv
- subgroup_performance_selected_threshold.csv
- subgroup_age_group_performance.png
- subgroup_gender_performance.png
- permutation_importance_average_precision.csv
- permutation_importance_average_precision.png

Notes:
- The saved logistic regression model is loaded and evaluated as-is.
- The inferential statsmodels model is an exploratory unpenalized logistic model
  fit on the training data for odds-ratio confidence intervals and p-values.
  It is not identical to the saved penalized, class-weighted sklearn model.
- The ROC-based threshold was previously selected on the test ROC curve and
  remains exploratory.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


TRAIN_PATH = "train_dataset.csv"
TEST_PATH = "test_dataset.csv"
MODEL_PATH = "logistic_regression_stroke_model.pkl"
TARGET = "stroke"


def classification_metrics(y_true, probabilities, threshold=0.50):
    """Return standard binary classification metrics at a selected threshold."""
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, pred, zero_division=0),
        "npv": npv,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc_avg_precision": average_precision_score(y_true, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ---------------------------------------------------------------------
# Load data and saved model
# ---------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
saved_model = joblib.load(MODEL_PATH)

id_columns = [c for c in ["id", "ID", "patient_id", "Patient_ID"] if c in train.columns]

X_train = train.drop(columns=[TARGET] + id_columns)
y_train = train[TARGET].astype(int)
X_test = test.drop(columns=[TARGET] + [c for c in id_columns if c in test.columns])
y_test = test[TARGET].astype(int)

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

test_probabilities = saved_model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------
# Calibration assessment
# ---------------------------------------------------------------------
test_roc_auc = roc_auc_score(y_test, test_probabilities)
test_pr_auc = average_precision_score(y_test, test_probabilities)
test_brier = brier_score_loss(y_test, test_probabilities)

fraction_positive, mean_predicted = calibration_curve(
    y_test, test_probabilities, n_bins=10, strategy="quantile"
)

pd.DataFrame(
    {
        "mean_predicted_probability": mean_predicted,
        "observed_stroke_rate": fraction_positive,
    }
).to_csv("calibration_curve_data.csv", index=False)

# Calibration intercept and slope from logistic regression of outcome on logit(p).
epsilon = 1e-6
logit_p = np.log(
    np.clip(test_probabilities, epsilon, 1 - epsilon)
    / (1 - np.clip(test_probabilities, epsilon, 1 - epsilon))
).reshape(-1, 1)

calibration_model = LogisticRegression(
    penalty=None, solver="lbfgs", max_iter=1000
).fit(logit_p, y_test)

calibration_intercept = float(calibration_model.intercept_[0])
calibration_slope = float(calibration_model.coef_[0][0])

plt.figure(figsize=(7, 5))
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.plot(mean_predicted, fraction_positive, marker="o", label="Saved logistic model")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed stroke rate")
plt.title("Test Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig("test_calibration_curve.png", dpi=200)
plt.close()

pd.DataFrame(
    [
        {
            "test_roc_auc": test_roc_auc,
            "test_pr_auc_avg_precision": test_pr_auc,
            "test_brier_score": test_brier,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
        }
    ]
).to_csv("recommended_improvements_summary.csv", index=False)


# ---------------------------------------------------------------------
# Exploratory inferential logistic regression for CIs and p-values
# ---------------------------------------------------------------------
try:
    one_hot = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
except TypeError:
    one_hot = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

stats_preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", one_hot),
                ]
            ),
            categorical_features,
        ),
    ]
)

X_train_stats = stats_preprocess.fit_transform(X_train)
stats_feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in stats_preprocess.get_feature_names_out()
]

X_train_stats = pd.DataFrame(X_train_stats, columns=stats_feature_names)
X_train_stats_const = sm.add_constant(X_train_stats, has_constant="add")

glm = sm.GLM(y_train.values, X_train_stats_const, family=sm.families.Binomial()).fit()

params = glm.params
conf_int = glm.conf_int()
inferential_results = pd.DataFrame(
    {
        "feature": params.index,
        "coefficient": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_lower_95": np.exp(conf_int[0].values),
        "ci_upper_95": np.exp(conf_int[1].values),
        "p_value": glm.pvalues.values,
    }
).sort_values("odds_ratio", ascending=False)

inferential_results.to_csv("statsmodels_logistic_or_ci_pvalues.csv", index=False)


# ---------------------------------------------------------------------
# Multicollinearity check with VIF
# ---------------------------------------------------------------------
X_vif = X_train_stats.loc[:, X_train_stats.std(axis=0) > 1e-12]
vif_rows = []

for index, feature in enumerate(X_vif.columns):
    vif_rows.append(
        {
            "feature": feature,
            "vif": variance_inflation_factor(X_vif.values, index),
        }
    )

pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
    "multicollinearity_vif.csv", index=False
)


# ---------------------------------------------------------------------
# Compare additional models without hyperparameter tuning
# ---------------------------------------------------------------------
tree_preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

benchmark_models = {
    "Saved tuned balanced logistic regression": saved_model,
    "Random forest balanced (untuned)": Pipeline(
        steps=[
            ("preprocess", tree_preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=5,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "Gradient boosting (untuned)": Pipeline(
        steps=[
            ("preprocess", tree_preprocess),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=50,
                    learning_rate=0.05,
                    max_depth=2,
                ),
            ),
        ]
    ),
}

comparison_rows = []

for model_name, candidate_model in benchmark_models.items():
    if model_name == "Saved tuned balanced logistic regression":
        probabilities = test_probabilities
    else:
        candidate_model.fit(X_train, y_train)
        probabilities = candidate_model.predict_proba(X_test)[:, 1]

    row = {"model": model_name, "threshold": 0.50}
    row.update(classification_metrics(y_test, probabilities, threshold=0.50))
    comparison_rows.append(row)

pd.DataFrame(comparison_rows).sort_values(
    "pr_auc_avg_precision", ascending=False
).to_csv("model_comparison_test_results.csv", index=False)


# ---------------------------------------------------------------------
# Subgroup/fairness-style performance analysis
# ---------------------------------------------------------------------
# This value was selected in the prior test-set ROC analysis and should be
# interpreted as exploratory.
prior_metrics = pd.read_csv("test_logistic_regression_metrics.csv")
selected_threshold = float(
    prior_metrics.loc[
        prior_metrics["threshold_label"].eq("ROC Youden J threshold"), "threshold"
    ].iloc[0]
)

subgroup_data = test.copy()
subgroup_data["predicted_probability"] = test_probabilities
subgroup_data["age_group"] = pd.cut(
    subgroup_data["age"],
    bins=[-np.inf, 18, 35, 50, 65, np.inf],
    labels=["<=18", "19-35", "36-50", "51-65", "66+"],
)

subgroup_rows = []

for subgroup_variable in ["Gender", "age_group", "Residence_type", "smoking_status"]:
    for subgroup_value, subgroup_df in subgroup_data.groupby(
        subgroup_variable, dropna=False, observed=False
    ):
        subgroup_pred = (
            subgroup_df["predicted_probability"].values >= selected_threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            subgroup_df[TARGET].astype(int), subgroup_pred, labels=[0, 1]
        ).ravel()

        subgroup_rows.append(
            {
                "group_variable": subgroup_variable,
                "group": str(subgroup_value),
                "threshold": selected_threshold,
                "n": len(subgroup_df),
                "stroke_cases": int(subgroup_df[TARGET].sum()),
                "stroke_rate": subgroup_df[TARGET].mean(),
                "accuracy": accuracy_score(subgroup_df[TARGET], subgroup_pred),
                "precision": tp / (tp + fp) if (tp + fp) else 0,
                "recall_sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "npv": tn / (tn + fn) if (tn + fn) else np.nan,
                "f1": f1_score(subgroup_df[TARGET], subgroup_pred, zero_division=0),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

subgroup_results = pd.DataFrame(subgroup_rows)
subgroup_results.to_csv("subgroup_performance_selected_threshold.csv", index=False)

for subgroup_variable, filename in [
    ("age_group", "subgroup_age_group_performance.png"),
    ("Gender", "subgroup_gender_performance.png"),
]:
    plot_df = subgroup_results[
        subgroup_results["group_variable"].eq(subgroup_variable)
    ].copy()

    x_positions = np.arange(len(plot_df))

    plt.figure(figsize=(8, 5))
    plt.bar(
        x_positions - 0.2,
        plot_df["recall_sensitivity"].fillna(0),
        width=0.4,
        label="Sensitivity",
    )
    plt.bar(
        x_positions + 0.2,
        plot_df["specificity"].fillna(0),
        width=0.4,
        label="Specificity",
    )
    plt.xticks(x_positions, plot_df["group"].astype(str), rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Metric value")
    plt.title(f"Subgroup Sensitivity and Specificity by {subgroup_variable}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Permutation importance using average precision as the scoring metric
# ---------------------------------------------------------------------
baseline_average_precision = average_precision_score(y_test, test_probabilities)
rng = np.random.default_rng(42)

importance_rows = []

for feature in X_test.columns:
    decreases = []

    for _ in range(3):
        X_permuted = X_test.copy()
        X_permuted[feature] = rng.permutation(X_permuted[feature].values)
        permuted_average_precision = average_precision_score(
            y_test, saved_model.predict_proba(X_permuted)[:, 1]
        )
        decreases.append(baseline_average_precision - permuted_average_precision)

    importance_rows.append(
        {
            "feature": feature,
            "importance_mean_pr_auc_decrease": float(np.mean(decreases)),
            "importance_std": float(np.std(decreases)),
        }
    )

importance_results = pd.DataFrame(importance_rows).sort_values(
    "importance_mean_pr_auc_decrease", ascending=False
)

importance_results.to_csv("permutation_importance_average_precision.csv", index=False)

plt.figure(figsize=(8, 5))
top_importance = importance_results.head(10).iloc[::-1]
plt.barh(
    top_importance["feature"],
    top_importance["importance_mean_pr_auc_decrease"],
)
plt.xlabel("Mean decrease in average precision")
plt.title("Permutation Importance on Test Set")
plt.tight_layout()
plt.savefig("permutation_importance_average_precision.png", dpi=200)
plt.close()
