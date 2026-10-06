install.packages("readxl")
library(readxl)
install.packages("caret")
library(caret)
install.packages("pROC")
library(pROC)

# Read data
mydata <- read_excel(file.choose())
# View first rows
head(mydata)
# Check structure
str(mydata)

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


# ----- LOGISTIC REG ---------------------------
# Model training & Time elapsed
start_time <- Sys.time()

mod_main <- glm(
  stroke ~ Gender +
    age +
    hypertension +
    heart_disease +
    ever_married +
    Residence_type +
    avg_glucose_level +
    bmi +
    smoking_status,
  
  data = train_data,
  family = binomial(link = "logit")
)

end_time <- Sys.time()
elapsed_time <- end_time - start_time
cat(sprintf(
  "Training time: %.4f seconds\n",
  as.numeric(elapsed_time, units = "secs")
))

# Model Summary Table
summary(mod_main)
# --------------------------------------------


# ----- LOGISTIC REG (with interaction terms) ----------------
# Model training & Time elapsed
start_time <- Sys.time()

mod_interaction <- glm(
  
  stroke ~
    Gender +    
    age*bmi +
    heart_disease+
    hypertension +
    
    bmi *avg_glucose_level,
  
  data = train_data,
  family = binomial
)

end_time <- Sys.time()
elapsed_time <- end_time - start_time
cat(sprintf(
  "Training time: %.4f seconds\n",
  as.numeric(elapsed_time, units = "secs")
))

# Model Summary Table
summary(mod_interaction)
# --------------------------------------------


# -------- ODD RATIOS ---------------
# Convert coefficients to odds ratios
exp(coef(mod_main))
exp(coef(mod_interaction))

# Confidence intervals for odds ratios
exp(confint(mod_main))
exp(confint(mod_interaction))

# Significance table (variable, odds ratio, p-value)
summary(mod_main)$coefficients
summary(mod_interaction)$coefficients
# ------------------------------------------------


# -------- MODEL COMPARISON -----------------------
# Compare models
AIC(mod_main)
AIC(mod_interaction) # smaller

# Likelihood Ratio Test
anova(mod_main, mod_interaction, test = "Chisq")
# --------------------------------------------


# ------ PREDICTED PROBABILITIES (Main) ---------------
# Predicted stroke probabilities 
test_data$test_probs_main <- predict(mod_main, newdata = test_data, type = "response")
# View first probabilities
head(test_probs)
# Age vs. Stroke Predicted Probability 
ggplot(
  data.frame(
    age = test_data$age,
    prob = test_probs
  ),
  aes(x = age, y = prob)
) +
  
  geom_point(alpha = 0.4) +
  
  geom_smooth(method = "loess") +
  
  labs(
    title = "Predicted Stroke Probability by Age",
    x = "Age",
    y = "Predicted Probability"
  )
# BMI vs. Stroke Predicted Probability 
ggplot(
  data.frame(
    bmi = test_data$bmi,
    prob = test_probs
  ),
  aes(x = bmi, y = prob)
) +
  
  geom_point(alpha = 0.4) +
  
  geom_smooth(method = "loess") +
  
  labs(
    title = "Predicted Stroke Probability by BMI",
    x = "BMI",
    y = "Predicted Probability"
  )
# --------------------------------------------------


# ------ PREDICTED PROBABILITIES (Interaction) ---------------
# Predicted stroke probabilities 
test_data$test_probs <- predict(mod_interaction, newdata = test_data, type = "response")
# View first probabilities
head(test_probs)
# Age vs. Stroke Predicted Probability 
ggplot(
  data.frame(
    age = test_data$age,
    prob = test_probs
  ),
  aes(x = age, y = prob)
) +
  
  geom_point(alpha = 0.4) +
  
  geom_smooth(method = "loess") +
  
  labs(
    title = "Predicted Stroke Probability by Age",
    x = "Age",
    y = "Predicted Probability"
)
# BMI vs. Stroke Predicted Probability 
ggplot(
  data.frame(
    bmi = test_data$bmi,
    prob = test_probs
  ),
  aes(x = bmi, y = prob)
) +
  
  geom_point(alpha = 0.4) +
  
  geom_smooth(method = "loess") +
  
  labs(
    title = "Predicted Stroke Probability by BMI",
    x = "AMI",
    y = "Predicted Probability"
)
# --------------------------------------------------


# ------------- ACCURACY TESTS -----------------------------------------------

# ------- THRESHOLD LOOP: 0.3, 0.4, 0.5, 0.6, 0.7 --------------
thresholds <- c(0.3, 0.4, 0.5, 0.6, 0.7)
results <- data.frame()

# Loop through threshold values
for (t in thresholds) {
  preds <- ifelse(test_probs > t, 1, 0)
  # Confusion Matrix
  conf <- table(
    factor(preds, levels = c(0,1)),
    factor(test_data$stroke, levels = c(0,1))
  )
  # Extract values
  TN <- conf["0","0"]
  TP <- conf["1","1"]
  FP <- conf["1","0"]
  FN <- conf["0","1"]
  
  accuracy <- mean(preds == test_data$stroke)
  # Sensitivity
  sensitivity <- TP / (TP + FN)
  # Specificity
  specificity <- TN / (TN + FP)
  # Store results
  results <- rbind(
    results,
    data.frame(
      Threshold = t,
      Accuracy = accuracy,
      Sensitivity = sensitivity,
      Specificity = specificity
    )
  )
}
# Return results
results
write.csv(results, "threshold_results.csv")
# -----------------------------------------------------------


# For one specific threshold:
test_preds <- ifelse(test_data$test_probs >  0.3867, 1, 0)
test_accuracy <- mean(test_preds == test_data$stroke)
# Confusion Matrix
conf_mat <- table(
    Predicted = test_preds,
    Actual = test_data$stroke
)
# Extract values
TN <- conf_mat["0", "0"]
TP <- conf_mat["1", "1"]
FP <- conf_mat["1", "0"]
FN <- conf_mat["0", "1"]

# Sensitivity
sensitivity <- TP / (TP + FN)
# Specificity
specificity <- TN / (TN + FP)
# Print results
cat(sprintf("Accuracy: %.4f\n", test_accuracy))
cat(sprintf("Sensitivity: %.4f\n", sensitivity))
cat(sprintf("Specificity: %.4f\n", specificity))

# ROC Curve
roc_curve <- roc(test_data$stroke, test_data$test_probs)
auc_value <- auc(roc_curve)
plot(
  roc_curve,
  main = paste(
    "ROC Curve (AUC =",
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
# -----------------------------------------------------------


# Convert matrix to dataframe
cm_df <- as.data.frame(as.table(conf_mat))

colnames(cm_df) <- c("Predicted", "Actual", "Freq")

# Better labels
cm_df$Predicted <- factor(
  cm_df$Predicted,
  levels = c(0,1),
  labels = c("Predicted No Stroke", "Predicted Stroke")
)

cm_df$Actual <- factor(
  cm_df$Actual,
  levels = c(0,1),
  labels = c("Actual No Stroke", "Actual Stroke")
)

# Heatmap
ggplot(cm_df,
       aes(x = Predicted,
           y = Actual,
           fill = Freq)) +
  
  geom_tile(color = "white") +
  
  geom_text(aes(label = Freq),
            size = 6) +
  
  scale_fill_gradient(
    low = "lightblue",
    high = "red"
  ) +
  
  labs(
    title = "Confusion Matrix for Stroke Prediction",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  
  theme_minimal(base_size = 14)

