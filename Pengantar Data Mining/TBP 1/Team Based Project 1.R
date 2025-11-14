# ==========================================
# Import Library
# ==========================================
library(tidyverse)
library(readr)
library(caret)
library(recipes)
library(rpart)
library(randomForest)
library(DT)
library(skimr)
library(knitr)
library(kableExtra)
library(tibble)
library(dplyr)
library(ggplot2)
library(forcats)

# ==========================================
# Helper Functions
# ==========================================
get_metrics <- function(actual, predicted) {
  actual    <- factor(actual, levels = c("No","Yes"))
  predicted <- factor(predicted, levels = c("No","Yes"))
  cm <- caret::confusionMatrix(predicted, actual, positive = "Yes")
  acc <- cm$overall["Accuracy"]
  recall <- cm$byClass["Recall"]
  precision <- ifelse(!is.na(cm$byClass["Precision"]),
                      cm$byClass["Precision"], cm$byClass["Pos Pred Value"])
  f1 <- ifelse(!is.na(cm$byClass["F1"]),
               cm$byClass["F1"],
               ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), NA_real_))
  tibble(
    Accuracy  = as.numeric(acc),
    Recall    = as.numeric(recall),
    Precision = as.numeric(precision),
    F1_Score  = as.numeric(f1)
  )
}

best_pred <- function(fit) {
  p <- fit$pred
  if (is.null(fit$bestTune) || ncol(fit$bestTune) == 0) return(p)
  idx <- rep(TRUE, nrow(p))
  for (nm in names(fit$bestTune)) {
    idx <- idx & (p[[nm]] == fit$bestTune[[nm]][1])
  }
  p[idx, , drop = FALSE]
}

mean_metrics_from_pred <- function(pred_df, label) {
  per_fold <- pred_df %>%
    group_by(Resample) %>%
    summarise(
      Accuracy  = get_metrics(obs, pred)$Accuracy,
      Recall    = get_metrics(obs, pred)$Recall,
      Precision = get_metrics(obs, pred)$Precision,
      F1_Score  = get_metrics(obs, pred)$F1_Score,
      .groups = "drop"
    )
  per_fold %>%
    summarise(
      Accuracy  = round(mean(Accuracy,  na.rm = TRUE), 4),
      Recall    = round(mean(Recall,    na.rm = TRUE), 4),
      Precision = round(mean(Precision, na.rm = TRUE), 4),
      F1_Score  = round(mean(F1_Score,  na.rm = TRUE), 4)
    ) %>%
    mutate(Model = label, .before = 1)
}

# ==========================================
# Data Loading
# ==========================================
set.seed(456)
data_raw <- read_csv("dataset.csv", na = c("", "NA", "nan"), show_col_types = FALSE)
if (!("Y" %in% names(data_raw))) stop("Kolom target 'Y' tidak ditemukan di dataset.")

DT::datatable(head(data_raw, 20),
              options = list(pageLength = 10, scrollX = TRUE),
              caption  = "Sampel Data (Raw)")

# ==========================================
# Konversi Semua Kolom ke Factor
# ==========================================
data_cat <- data_raw %>%
  mutate(across(everything(), as.factor)) %>%
  mutate(Y = factor(Y, levels = c("0", "1"), labels = c("No", "Yes")))

# ==========================================
# EDA: Plot Setiap Variabel
# ==========================================
cat_cols <- names(data_cat)
for (col in cat_cols) {
  print(
    ggplot(data_cat, aes(x = fct_infreq(.data[[col]]))) +
      geom_bar(fill = "#3498DB") +
      coord_flip() +
      labs(title = paste("Distribusi Variabel:", col),
           x = col, y = "Jumlah Observasi") +
      theme_minimal(base_size = 13)
  )
}

# Plot distribusi target secara eksplisit
ggplot(data_cat, aes(x = Y)) +
  geom_bar(fill = "#2ECC71") +
  labs(title = "Distribusi Variabel Target (Y)",
       x = "Kupon Diterima", y = "Jumlah Observasi") +
  theme_minimal(base_size = 13)

# ==========================================
# Persiapan Data untuk Modeling
# ==========================================
data_cat <- data_cat %>% select(-any_of(c("car", "direction_opp", "toCoupon_GEQ5min")))

# ==========================================
# Recipe
# ==========================================
rec_base <- recipe(Y ~ ., data = data_cat) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors(), new_level = "new") %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# ==========================================
# Cross Validation
# ==========================================
set.seed(456)
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  savePredictions = "final",
  verboseIter = FALSE,
  allowParallel = TRUE
)

# ==========================================
# Train Models (LR, DT, RF)
# ==========================================
set.seed(456)
fit_lr <- train(rec_base, data = data_cat, method = "glm", family = binomial(), trControl = ctrl)
set.seed(456)
fit_dt <- train(rec_base, data = data_cat, method = "rpart", trControl = ctrl)
set.seed(456)
fit_rf <- train(rec_base, data = data_cat, method = "rf", trControl = ctrl)

# ==========================================
# Confusion Matrix
# ==========================================
pred_lr  <- best_pred(fit_lr)
pred_dt  <- best_pred(fit_dt)
pred_rf  <- best_pred(fit_rf)

cm_lr <- confusionMatrix(pred_lr$pred, pred_lr$obs, positive = "Yes")
cm_dt <- confusionMatrix(pred_dt$pred, pred_dt$obs, positive = "Yes")
cm_rf <- confusionMatrix(pred_rf$pred, pred_rf$obs, positive = "Yes")

cat("=== Confusion Matrix: Logistic Regression ===\n")
print(cm_lr$table)
cat("\nAkurasi:", round(cm_lr$overall["Accuracy"], 4), "\n\n")

cat("=== Confusion Matrix: Decision Tree ===\n")
print(cm_dt$table)
cat("\nAkurasi:", round(cm_dt$overall["Accuracy"], 4), "\n\n")

cat("=== Confusion Matrix: Random Forest ===\n")
print(cm_rf$table)
cat("\nAkurasi:", round(cm_rf$overall["Accuracy"], 4), "\n\n")

# ==========================================
# Rata-rata 10-Fold CV
# ==========================================
mean_lr <- mean_metrics_from_pred(pred_lr, "Logistic Regression")
mean_dt <- mean_metrics_from_pred(pred_dt, "Decision Tree")
mean_rf <- mean_metrics_from_pred(pred_rf, "Random Forest")

cv_mean_tbl <- bind_rows(mean_lr, mean_dt, mean_rf) %>%
  select(Model, Accuracy, Recall, Precision, F1_Score)

cv_mean_tbl %>%
  kable(align = "c", booktabs = TRUE,
        caption = "Rata-rata 10-Fold CV — Logistic Regression, Decision Tree, dan Random Forest") %>%
  kable_styling(full_width = FALSE, position = "center")

# ==========================================
# Feature Importance Random Forest
# ==========================================
rf_imp <- varImp(fit_rf)$importance %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall))

print(head(rf_imp, 15))

ggplot(rf_imp[1:15,], aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "#E67E22") +
  coord_flip() +
  labs(x = "Fitur", y = "Kepentingan (Importance Score)") +
  theme_minimal(base_size = 13)
