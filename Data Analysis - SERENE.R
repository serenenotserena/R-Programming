library(readxl)      
library(DataExplorer)
library(ggplot2)
library(caret)
library(caTools)
library(randomForest)
library(pROC)

df <- as.data.frame(read_excel("~/RStudio/employee_attrition.xlsx"))


# Clean TotalWorkingYears (Fixed: preserve minus sign first so negatives aren't turned positive)
df$TotalWorkingYears <- gsub("[^0-9.-]", "", as.character(df$TotalWorkingYears))
df$TotalWorkingYears <- as.numeric(df$TotalWorkingYears)
df$TotalWorkingYears[df$TotalWorkingYears < 0] <- NA  

# Clean Attrition (Fixed: added trimws() to strip accidental spaces)
df$Attrition <- trimws(tolower(as.character(df$Attrition)))
df$Attrition[df$Attrition == "0"] <- "no"
df$Attrition[df$Attrition == "1"] <- "yes"
df <- df[df$Attrition %in% c("yes", "no") & !is.na(df$Attrition), ]

# Clean Department (Fixed: added trimws() and converted blanks to NA cleanly)
df$Department <- trimws(toupper(as.character(df$Department)))
df$Department[df$Department == ""] <- NA
df$Department[df$Department == "HR"] <- "HUMAN RESOURCES"
df$Department[df$Department == "R&D"] <- "RESEARCH & DEVELOPMENT"
df$Department[df$Department == "SALE"] <- "SALES"

getMode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Create a temporary copy with imputed values strictly for EDA & Hypothesis Testing
df_eda <- df
df_eda$Department[is.na(df_eda$Department)] <- getMode(df_eda$Department)
df_eda$TotalWorkingYears[is.na(df_eda$TotalWorkingYears)] <- median(df_eda$TotalWorkingYears, na.rm = TRUE)

table(df_eda$Department)

# Plot 1: Attrition Distribution (Fixed: typo "Attrition" and lowercase "y")
ggplot(df_eda, aes(x = Attrition)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Employee Attrition Distribution",
       x = "Attrition",
       y = "Count")

# Analyze Department vs Attrition
table(df_eda$Department, df_eda$Attrition)
prop.table(table(df_eda$Department, df_eda$Attrition), 1)

ggplot(df_eda, aes(x = Department, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(title = "Attrition by Department",
       x = "Department",
       y = "Proportion")

# Analyze TotalWorkingYears vs Attrition
ggplot(df_eda, aes(x = Attrition,
                   y = TotalWorkingYears,
                   color = Attrition)) +
  geom_boxplot() +
  labs(title = "Total Working Years vs Attrition",
       x = "Attrition",
       y = "Total Working Years")

# Summary Statistics
aggregate(TotalWorkingYears ~ Attrition,
          data = df_eda,
          summary)

## Hypothesis Testing 

# Chi-Square Test: Department vs Attrition
tab <- table(df_eda$Department, df_eda$Attrition)
chisq <- chisq.test(tab)
print(chisq)

# T-Test: TotalWorkingYears vs Attrition
t_res <- t.test(TotalWorkingYears ~ Attrition, data = df_eda)
print(t_res)

##Supervised Learning (Leak-Free Pipeline)

df2 <- df[, c("Department", "TotalWorkingYears", "Attrition")]

# Convert Attrition target to factor
df2$Attrition <- factor(
  df2$Attrition,
  levels = c("no", "yes"),
  labels = c("0", "1")
)

# Train-Test Split BEFORE imputation to prevent data leakage
set.seed(123)
split <- sample.split(df2$Attrition, SplitRatio = 0.8)
training_set <- subset(df2, split == TRUE)
test_set <- subset(df2, split == FALSE)

# Impute missing values using ONLY training set statistics
train_dept_mode <- getMode(training_set$Department)
train_years_med <- median(training_set$TotalWorkingYears, na.rm = TRUE)

training_set$Department[is.na(training_set$Department)] <- train_dept_mode
test_set$Department[is.na(test_set$Department)] <- train_dept_mode

training_set$TotalWorkingYears[is.na(training_set$TotalWorkingYears)] <- train_years_med
test_set$TotalWorkingYears[is.na(test_set$TotalWorkingYears)] <- train_years_med

dept_levels <- unique(training_set$Department)
training_set$Department <- factor(training_set$Department, levels = dept_levels)
test_set$Department <- factor(test_set$Department, levels = dept_levels)

str(training_set)
table(training_set$Attrition)

# 1. Logistic Regression
classifier <- glm(
  Attrition ~ Department + TotalWorkingYears,
  data = training_set,
  family = "binomial"
)
summary(classifier)

pred_prob <- predict(classifier, type = "response", newdata = test_set)
pred_class <- factor(ifelse(pred_prob > 0.5, "1", "0"), levels = c("0", "1"))

# Fixed: Explicitly set positive = "1"
lr_cm <- confusionMatrix(pred_class, test_set$Attrition, positive = "1")
print(lr_cm)

# 2. Random Forest
rf_model <- randomForest(
  Attrition ~ Department + TotalWorkingYears,
  data = training_set,
  ntree = 500
)

rf_pred <- predict(rf_model, newdata = test_set)

# Fixed: Explicitly set positive = "1"
rf_cm <- confusionMatrix(rf_pred, test_set$Attrition, positive = "1")
print(rf_cm)

varImpPlot(rf_model)

# 3. ROC & AUC Comparison
lr_roc <- roc(test_set$Attrition, pred_prob, levels = c("0", "1"), direction = "<")
lr_auc <- as.numeric(auc(lr_roc))

rf_prob <- predict(rf_model, test_set, type = "prob")[, "1"]
rf_roc <- roc(test_set$Attrition, rf_prob, levels = c("0", "1"), direction = "<")
rf_auc <- as.numeric(auc(rf_roc))

plot(lr_roc, col = "gold", lwd = 2, main = "ROC Curve Comparison")
lines(rf_roc, col = "hotpink", lwd = 2)
legend("bottomright",
       legend = c(sprintf("Logistic Regression (AUC = %.3f)", lr_auc),
                  sprintf("Random Forest (AUC = %.3f)", rf_auc)),
       col = c("gold", "hotpink"),
       lwd = 2)

comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  AUC = c(lr_auc, rf_auc)
)
print(comparison)
