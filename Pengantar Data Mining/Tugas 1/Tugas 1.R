# Import Library
library(tidyverse)
library(readxl)
library(gridExtra)
library(corrplot)
library(caTools)
library(caret)
library(MLmetrics)
library(class)
library(rpart)
library(rpart.plot)
library(DT)
library(skimr)
library(knitr)
library(kableExtra)
library(tibble)
library(ggplot2)

# Helper Function
get_metrics <- function(actual, predicted, model_name, split_name) {
  cm <- caret::confusionMatrix(predicted, actual, positive = "Yes")
  acc <- cm$overall["Accuracy"]
  recall <- cm$byClass["Recall"]
  precision <- ifelse(!is.na(cm$byClass["Precision"]),
                      cm$byClass["Precision"], cm$byClass["Pos Pred Value"])
  f1 <- ifelse(!is.na(cm$byClass["F1"]),
               cm$byClass["F1"],
               2 * precision * recall / (precision + recall))
  data.frame(Model = model_name,
             Split = split_name,
             Accuracy = round(acc,4),
             Recall = round(recall,4),
             Precision = round(precision,4),
             F1_Score = round(f1,4))
}

# Load Data
set.seed(123)
data <- read_excel("dataset.xlsx")
print(head(data))

# Struktur data & NA
type_info <- tibble(
  Variabel = names(data),
  Tipe     = sapply(data, function(x) class(x)[1]),
  N_NA     = sapply(data, function(x) sum(is.na(x))),
  Prop_NA  = round(sapply(data, function(x) mean(is.na(x))), 4)
)
print(type_info)
skim(data)

# Plot EDA
p1 <- ggplot(data, aes(x = factor(`Gallstone Status`))) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Gallstone Status") +
  theme_minimal()

p2 <- ggplot(data, aes(x = Age)) +
  geom_histogram(fill = "lightblue", color = "black") +
  labs(title = "Age") +
  theme_minimal()

gridExtra::grid.arrange(p1, p2, ncol = 2)

# Convert tipe data
data$`Gallstone Status` <- as.factor(data$`Gallstone Status`)
data$`Hepatic Fat Accumulation (HFA)` <- as.factor(data$`Hepatic Fat Accumulation (HFA)`)
data$Gender <- as.factor(data$Gender)
data$Comorbidity <- as.factor(data$Comorbidity)
data$`Coronary Artery Disease (CAD)` <- as.factor(data$`Coronary Artery Disease (CAD)`)
data$Hypothyroidism <- as.factor(data$Hypothyroidism)
data$Hyperlipidemia <- as.factor(data$Hyperlipidemia)
data$`Diabetes Mellitus (DM)` <- as.factor(data$`Diabetes Mellitus (DM)`)

print(str(data))

# Missing Values
missing_tbl <- tibble(
  Variabel = names(data),
  N_Missing = sapply(data, function(x) sum(is.na(x)))
) %>% arrange(desc(N_Missing))
print(missing_tbl)

# Duplikasi
n_dups <- sum(duplicated(data))
cat("Jumlah duplikasi:", n_dups, "\n")

# Splitting
set.seed(123)
data_clean <- data %>%
  mutate(`Gallstone Status` = factor(`Gallstone Status`,
                                     levels = c(0,1),
                                     labels = c("No","Yes")))
idx <- createDataPartition(data_clean$`Gallstone Status`, p = 0.8, list = FALSE)
train <- data_clean[idx, ]
test  <- data_clean[-idx, ]

# Scaling
num_cols <- names(train)[sapply(train, is.numeric)]

sc_train <- scale(train[, num_cols])
center_vec <- attr(sc_train, "scaled:center")
scale_vec <- attr(sc_train, "scaled:scale")

train_scaled <- train
train_scaled[, num_cols] <- sc_train

test_scaled <- test
test_scaled[, num_cols] <- scale(test[, num_cols],
                                 center = center_vec,
                                 scale = scale_vec)

# Decision Tree
fit_dt <- rpart(`Gallstone Status` ~ ., data = train_scaled, method = "class")
rpart.plot(fit_dt)

pred_dt_train <- predict(fit_dt, newdata = train_scaled, type = "class")
metrics_dt_train <- get_metrics(train$`Gallstone Status`, pred_dt_train,
                                "Decision Tree", "Train")

pred_dt_test <- predict(fit_dt, newdata = test_scaled, type = "class")
metrics_dt_test <- get_metrics(test$`Gallstone Status`, pred_dt_test,
                               "Decision Tree", "Test")

# Logistic Regression
fit_lr <- glm(`Gallstone Status` ~ ., data = train_scaled, family = binomial)

pred_lr_train_prob <- predict(fit_lr, newdata = train_scaled, type = "response")
pred_lr_train <- factor(ifelse(pred_lr_train_prob >= 0.5, "Yes","No"),
                        levels = c("No","Yes"))
metrics_lr_train <- get_metrics(train_scaled$`Gallstone Status`, pred_lr_train,
                                "Logistic Regression", "Train")

pred_lr_test_prob <- predict(fit_lr, newdata = test_scaled, type = "response")
pred_lr_test <- factor(ifelse(pred_lr_test_prob >= 0.5, "Yes","No"),
                       levels = c("No","Yes"))
metrics_lr_test <- get_metrics(test_scaled$`Gallstone Status`, pred_lr_test,
                               "Logistic Regression", "Test")

# Ensemble (average probability)
pred_dt_prob_train <- predict(fit_dt, newdata = train_scaled, type = "prob")[,"Yes"]
pred_lr_prob_train <- predict(fit_lr, newdata = train_scaled, type = "response")
pred_ens_train <- factor(
  ifelse((pred_dt_prob_train + pred_lr_prob_train)/2 >= 0.5, "Yes","No"),
  levels = c("No","Yes")
)
metrics_ens_train <- get_metrics(train_scaled$`Gallstone Status`, pred_ens_train,
                                 "Ensemble", "Train")

pred_dt_prob_test <- predict(fit_dt, newdata = test_scaled, type = "prob")[,"Yes"]
pred_lr_prob_test <- predict(fit_lr, newdata = test_scaled, type = "response")
pred_ens_test <- factor(
  ifelse((pred_dt_prob_test + pred_lr_prob_test)/2 >= 0.5, "Yes","No"),
  levels = c("No","Yes")
)
metrics_ens_test <- get_metrics(test_scaled$`Gallstone Status`, pred_ens_test,
                                "Ensemble", "Test")

# Ringkasan Semua Model
metrics_all <- bind_rows(metrics_dt_train, metrics_dt_test,
                         metrics_lr_train, metrics_lr_test,
                         metrics_ens_train, metrics_ens_test)

print(metrics_all)

# Plot Perbandingan
ggplot(metrics_all %>%
         pivot_longer(cols = c(Accuracy, Recall, Precision, F1_Score),
                      names_to = "Metric", values_to = "Value"),
       aes(x = Model, y = Value, fill = Split)) +
  geom_col(position = "dodge") +
  facet_wrap(~ Metric, scales = "free_y") +
  theme_minimal() +
  labs(title = "Perbandingan Metrik Model",
       x = "Model", y = "Nilai")
