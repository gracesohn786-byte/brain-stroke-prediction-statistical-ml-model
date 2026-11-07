# RANDOM FOREST MODEL FOR STROKE PREDICTION

# -------------------- LIBRARIES --------------------
install.packages("randomForest")
install.packages("caret")
install.packages("ggplot2")
install.packages("readxl")
install.packages("pROC")

library(randomForest)
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


# ----------- TRAIN / TEST SPLIT ---------------------
train_indices <- read_excel(file.choose())

# Convert first column to numeric vector
train_indices <- train_indices[[1]]

train_data <- mydata[train_indices, ]
test_data <- mydata[-train_indices, ]

cat("Training rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")

table(train_data$stroke)
table(test_data$stroke)

# Stroke rate in training dataset
train_stroke_rate <- mean(as.numeric(as.character(train_data$stroke)))

# Stroke rate in testing dataset
test_stroke_rate <- mean(as.numeric(as.character(test_data$stroke)))

cat("Training Stroke Rate:",
    round(train_stroke_rate * 100, 2),
    "%\n")
cat("Testing Stroke Rate:",
    round(test_stroke_rate * 100, 2),
    "%\n")
# ---------------------------------------------------


# -------------- RANDOM FOREST MODEL (100 trees) ----------------
train_data$stroke <- as.factor(train_data$stroke)
start_time <- Sys.time()

forest_model <- randomForest(
  stroke ~ .,
  data = train_data,
  ntree = 100,
  importance = TRUE,
  classwt = c("0" = 1,
              "1" = 19)
)

end_time <- Sys.time()
elapsed_time <- as.numeric(
  difftime(end_time, start_time, units = "secs")
)
cat(
  "Training Time:",
  round(elapsed_time, 4),
  "seconds\n"
)
# ---------------------------------------------------


# ---------------- MODEL SUMMARY ----------------
print(forest_model)
# ---------------------------------------------------

# -------------- VARIABLE IMPORTANCE --------------------
importance(forest_model)
varImpPlot(
  forest_model,
  main = "Variable Importance - Random Forest"
)
# ---------------------------------------------------


# ---------------- PREDICTIONS ------------------
# Predicted classes
pred_rf <- predict(
  forest_model,
  newdata = test_data
)

# Predicted probabilities
pred_probs <- predict(
  forest_model,
  newdata = test_data,
  type = "prob"
)
# View first probabilities
head(pred_probs[,2])
# ---------------------------------------------------


# --------------- ACCURACY --------------------
accuracy <- mean(
  pred_rf == test_data$stroke
)
cat(
  "Test Accuracy:",
  round(accuracy, 4),
  "\n"
)
# ---------------------------------------------------


# -------------- CONFUSION MATRIX --------------------
conf_mat <- table(
  Predicted = pred_rf,
  Actual = test_data$stroke
)
print(conf_mat)
# ---------------------------------------------------


# ------------ SENSITIVITY & SPECIFICITY ---------------
TN <- conf_mat["0", "0"]
TP <- conf_mat["1", "1"]
FP <- conf_mat["1", "0"]
FN <- conf_mat["0", "1"]

sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)

cat(
  "Sensitivity:",
  round(sensitivity, 4),
  "\n"
)
cat(
  "Specificity:",
  round(specificity, 4),
  "\n"
)
# --------------------------------------------------


# -------------- ROC CURVE & AUC ------------------
roc_curve <- roc(
  test_data$stroke,
  pred_probs[,2]
)
best <- coords(
  roc_curve,
  "best",
  ret = c("threshold","sensitivity","specificity"),
  best.method = "youden"
)

print(best)

threshold <- best$threshold

pred_rf <- ifelse(
  pred_probs[,2] >= threshold,
  1,
  0
)

pred_rf <- factor(pred_rf, levels = c(0,1))

auc_value <- auc(roc_curve)

plot(
  roc_curve,
  main = paste(
    "Random Forest ROC Curve (AUC =",
    round(auc_value, 3),
    ")"
  )
)
# ---------------------------------------------------


# ------------- TRAIN VS TEST ACCURACY ---------------------
# Training predictions
train_pred <- predict(
  forest_model,
  newdata = train_data
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


# ----------- OOB Score ------------------------
final_oob <- forest_model$err.rate[
  nrow(forest_model$err.rate),
  "OOB"
]

cat("Final OOB Error:", round(final_oob * 100, 2), "%\n")
# ---------------------------------------------------


# ----------- ERROR RATE PLOT -------------------
plot(
  forest_model,
  main = "Random Forest Error Rate"
)
# ---------------------------------------------------

# ----------- OPTIONAL: SAVE RESULTS -----------------
# write.csv(
#   cm_df,
#   "random_forest_confusion_matrix.csv"
# )


# ------------- CONFUSION MATRIX HEATMAP -------------------
cm_df <- as.data.frame(as.table(conf_mat))

colnames(cm_df) <- c(
  "Predicted",
  "Actual",
  "Freq"
)

# Better labels
cm_df$Predicted <- factor(
  cm_df$Predicted,
  levels = c("0", "1"),
  labels = c(
    "Predicted No Stroke",
    "Predicted Stroke"
  )
)
cm_df$Actual <- factor(
  cm_df$Actual,
  levels = c("0", "1"),
  labels = c(
    "Actual No Stroke",
    "Actual Stroke"
  )
)

# Heatmap
ggplot(
  cm_df,
  aes(
    x = Predicted,
    y = Actual,
    fill = Freq
  )
) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = Freq),
    size = 6
  ) +
  scale_fill_gradient(
    low = "lightblue",
    high = "darkblue"
  ) +
  labs(
    title = "Confusion Matrix for Random Forest",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  
  theme_minimal(base_size = 14)
# ---------------------------------------------------

