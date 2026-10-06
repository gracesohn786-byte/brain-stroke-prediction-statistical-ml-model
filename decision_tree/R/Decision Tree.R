# DECISION TREE MODEL

# -------------------- LIBRARIES --------------------
install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")
install.packages("ggplot2")
install.packages("readxl")
install.packages("pROC")

library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(readxl)
library(pROC)

# ---------------------------------------------------

# Read data
mydata <- read_excel(file.choose())

# Convert categorical variables to factors
mydata$Gender <- as.factor(mydata$Gender)
mydata$ever_married <- as.factor(mydata$ever_married)
mydata$work_type <- as.factor(mydata$work_type)
mydata$Residence_type <- as.factor(mydata$Residence_type)
mydata$smoking_status <- as.factor(mydata$smoking_status)
mydata$hypertension <- as.factor(mydata$hypertension)
mydata$heart_disease <- as.factor(mydata$heart_disease)


# Train/Test Split (50% Training, 50% Testing), Evenly divide stroke positives
set.seed(42)

train_indices <- createDataPartition(
  mydata$stroke,
  p = 0.5,
  list = FALSE
)

train_data <- mydata[train_indices, ]
test_data  <- mydata[-train_indices, ]
# ---------------------------------------------------


# -------------- Model #1 Original -----------------------
start_time <- Sys.time()

tree_model_original <- rpart(
  
  stroke ~
    Gender +
    age +
    hypertension +
    heart_disease +
    ever_married +
    work_type +
    Residence_type +
    avg_glucose_level +
    bmi +
    smoking_status,
  
  data = train_data,
  method = "class"
)

end_time <- Sys.time()
elapsed_time <- as.numeric(
  difftime(end_time, start_time, units = "secs")
)
cat("Training Time:", round(elapsed_time, 4), "seconds\n")
# MODEL SUMMARY
summary(tree_model_original)

# Variable importance
tree_model_original$variable.importance
# ---------------------------------------------------


# ---------- Model #3 Lower minimum split restrictions -------------
start_time <- Sys.time()

tree_model_tuned <- rpart(
  stroke ~ .,
  data = train_data,
  method = "class",
  control = rpart.control(
    minsplit = 10,
    cp = 0.0001,
    maxdepth = 5
  )
)
end_time <- Sys.time()
elapsed_time <- as.numeric(
  difftime(end_time, start_time, units = "secs")
)
cat("Training Time:", round(elapsed_time, 4), "seconds\n")
# MODEL SUMMARY
summary(tree_model_tuned)

# Variable importance
tree_model_tuned$variable.importance
# ---------------------------------------------------


# ---------- Model #2 CLASS BALANCING ---------------------------
# Create balanced training data
stroke_cases <- train_data[train_data$stroke == 1, ]
nonstroke_cases <- train_data[train_data$stroke == 0, ]

set.seed(42)

nonstroke_sample <- nonstroke_cases[
  sample(
    1:nrow(nonstroke_cases),
    nrow(stroke_cases)
  ),
]

balanced_train <- rbind(
  stroke_cases,
  nonstroke_sample
)
# Build tree
tree_model_balanced <- rpart(
  stroke ~ .,
  data = balanced_train,
  method = "class"
)
# Model Summary
summary(tree_model_balanced)
# Variable importance
tree_model_balanced$variable.importance
# ---------------------------------------------------


# ------------- PLOT DECISION TREE -----------------------
# Simple tree
rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  box.palette = "RdBu",
  shadow.col = "gray",
  main = "Decision Tree for Stroke Prediction"
)
# ---------------------------------------------------


# ---------------- PREDICTIONS --------------------
# Predicted classes
pred_tree <- predict(
  tree_model_tuned,
  newdata = test_data,
  type = "class"
)

# Predicted probabilities
pred_probs <- predict(
  tree_model_tuned,
  newdata = test_data,
  type = "prob"
)

# Lower classification threshold
threshold <-  0.04865

# Convert probabilities into class predictions
pred_tree <- ifelse(
  pred_probs[,2] > threshold,
  1,
  0
)

# Convert to factor
pred_tree <- factor(pred_tree, levels = c(0,1))
# ---------------------------------------------------


# ---------------- ACCURACY --------------------
accuracy <- mean(pred_tree == test_data$stroke)
cat("Test Accuracy:", round(accuracy, 4), "\n")
# ---------------------------------------------------

# -------------- CONFUSION MATRIX ------------------
conf_mat <- table(
  Predicted = pred_tree,
  Actual = test_data$stroke
)
print(conf_mat)
# ---------------------------------------------------

# -------------- SENSITIVITY & SPECIFICITY ---------------
TN <- conf_mat["0", "0"]
TP <- conf_mat["1", "1"]
FP <- conf_mat["1", "0"]
FN <- conf_mat["0", "1"]

sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)

cat("Sensitivity:", round(sensitivity, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")
# ---------------------------------------------------

# ---------------- ROC CURVE & AUC ---------------------
roc_curve <- roc(
  test_data$stroke,
  pred_probs[,2]
)

auc_value <- auc(roc_curve)

plot(
  roc_curve,
  main = paste(
    "Decision Tree ROC Curve (AUC =",
    round(auc_value, 3),
    ")"
  )
)

# Best threshold using Youden Index
best <- coords(
  roc_curve,
  "best",
  ret = c("threshold", "sensitivity", "specificity"),
  best.method = "youden"
)

print(best)
# ---------------------------------------------------

# -------------- TRAIN VS TEST ACCURACY ------------------
# Training predictions
train_pred <- predict(
  tree_model,
  newdata = train_data,
  type = "class"
)
train_accuracy <- mean(
  train_pred == train_data$stroke
)
cat(
  "Train Accuracy:",
  round(train_accuracy, 4),
  "\n"
)
cat(
  "Test Accuracy:",
  round(accuracy, 4),
  "\n"
)
# ---------------------------------------------------

# ----------------- VARIABLE IMPORTANCE PLOT -----------------
importance_df <- data.frame(
  Variable = names(tree_model_tuned$variable.importance),
  Importance = tree_model_tuned$variable.importance
)

ggplot(
  importance_df,
  aes(
    x = reorder(Variable, Importance),
    y = Importance
  )
) +
  
  geom_bar(stat = "identity") +
  
  coord_flip() +
  
  labs(
    title = "Variable Importance in Decision Tree",
    x = "Variables",
    y = "Importance"
  ) +
  
  theme_minimal()

# Export train and test datasets
write.csv(train_indices,
          "train_indices.csv",
          row.names = FALSE)