# =============================================================================
# Football Match Outcome Prediction — Classification
# Group 8 | Reproducible Research | University of Warsaw 2026
# =============================================================================
# Author  : Ashutosh Kumar Verma (475852)
# Issue   : #5 — Data Loading, Cleaning & EDA
# Dataset : ESPN Soccer Data 2024-2026 (Kaggle)
# =============================================================================

library(here)
library(tidyverse)
library(lubridate)
library(ggplot2)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
cat("Loading dataset...\n")

df <- read_csv(here("classification", "data", "fixtures.csv"))

cat("Dataset shape:", nrow(df), "rows x", ncol(df), "columns\n")
cat("Expected: 67,353 rows\n\n")

# -----------------------------------------------------------------------------
# 2. CREATE OUTCOME VARIABLE
# -----------------------------------------------------------------------------
df <- df %>%
  mutate(
    outcome = case_when(
      homeTeamWinner == TRUE ~ "Home Win",
      awayTeamWinner == TRUE ~ "Away Win",
      TRUE                   ~ "Draw"
    )
  )

cat("Outcome variable created:\n")
print(table(df$outcome))
cat("\n")

# -----------------------------------------------------------------------------
# 3. DROP RESULT-LEAKAGE COLUMNS
# -----------------------------------------------------------------------------
leakage_cols <- c(
  "homeTeamWinner", "awayTeamWinner",
  "homeTeamScore", "awayTeamScore",
  "homeTeamShootoutScore", "awayTeamShootoutScore",
  "statusId"
)

leakage_cols <- leakage_cols[leakage_cols %in% names(df)]
df <- df %>% select(-all_of(leakage_cols))

cat("Dropped leakage columns:", paste(leakage_cols, collapse = ", "), "\n")
cat("Remaining columns:", ncol(df), "\n\n")

# -----------------------------------------------------------------------------
# 4. PARSE DATES & SORT
# -----------------------------------------------------------------------------
df <- df %>%
  mutate(date = ymd_hms(date, quiet = TRUE)) %>%
  arrange(date)

cat("Date range:",
    format(min(df$date, na.rm = TRUE)),
    "to",
    format(max(df$date, na.rm = TRUE)), "\n\n")

# -----------------------------------------------------------------------------
# 5. MISSING VALUES CHECK
# -----------------------------------------------------------------------------
cat("Checking missing values...\n")
missing_summary <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(),
               names_to = "column",
               values_to = "missing") %>%
  filter(missing > 0)

if (nrow(missing_summary) == 0) {
  cat("No missing values found (matches Python baseline)\n\n")
} else {
  cat("Missing values found:\n")
  print(missing_summary)
}

# -----------------------------------------------------------------------------
# 6. CLASS DISTRIBUTION
# -----------------------------------------------------------------------------
cat("Class Distribution:\n")
class_dist <- df %>%
  count(outcome) %>%
  mutate(percentage = round(n / sum(n) * 100, 2))
print(class_dist)
cat("\n")

# -----------------------------------------------------------------------------
# 7. EDA VISUALISATIONS
# -----------------------------------------------------------------------------
dir.create(here("classification", "output"), showWarnings = FALSE)

# Plot 1 — Outcome distribution
p1 <- ggplot(df, aes(x = outcome, fill = outcome)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)),
            vjust = -0.5, size = 4) +
  scale_fill_manual(values = c(
    "Home Win" = "#2196F3",
    "Draw"     = "#FF9800",
    "Away Win" = "#F44336"
  )) +
  labs(
    title    = "Football Match Outcome Distribution",
    subtitle = paste("Total matches:", nrow(df)),
    x        = "Outcome",
    y        = "Count"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(here("classification", "output", "outcome_distribution.png"),
       p1, width = 8, height = 6, dpi = 300)
cat("Saved: outcome_distribution.png\n")

# Plot 2 — Outcomes by league (top 10)
if ("leagueId" %in% names(df)) {
  top_leagues <- df %>%
    count(leagueId) %>%
    slice_max(n, n = 10) %>%
    pull(leagueId)
  
  p2 <- df %>%
    filter(leagueId %in% top_leagues) %>%
    count(leagueId, outcome) %>%
    group_by(leagueId) %>%
    mutate(pct = n / sum(n) * 100) %>%
    ggplot(aes(x = factor(leagueId), y = pct, fill = outcome)) +
    geom_col(position = "stack") +
    scale_fill_manual(values = c(
      "Home Win" = "#2196F3",
      "Draw"     = "#FF9800",
      "Away Win" = "#F44336"
    )) +
    labs(
      title = "Match Outcomes by League (Top 10)",
      x     = "League ID",
      y     = "Percentage (%)",
      fill  = "Outcome"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(here("classification", "output", "outcomes_by_league.png"),
         p2, width = 10, height = 6, dpi = 300)
  cat("Saved: outcomes_by_league.png\n")
}

# Plot 3 — Temporal trends by year
if (!all(is.na(df$date))) {
  p3 <- df %>%
    mutate(year = year(date)) %>%
    count(year, outcome) %>%
    ggplot(aes(x = year, y = n, color = outcome, group = outcome)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    scale_color_manual(values = c(
      "Home Win" = "#2196F3",
      "Draw"     = "#FF9800",
      "Away Win" = "#F44336"
    )) +
    labs(
      title  = "Match Outcomes Over Time",
      x      = "Year",
      y      = "Number of Matches",
      color  = "Outcome"
    ) +
    theme_minimal()
  
  ggsave(here("classification", "output", "temporal_trends.png"),
         p3, width = 10, height = 6, dpi = 300)
  cat("Saved: temporal_trends.png\n")
}

cat("\n✓ Issue #5 complete — EDA outputs saved to classification/output/\n")

# =============================================================================
# Issue #6 — Feature Engineering & Preprocessing
# Author  : Ashutosh Kumar Verma (475852)
# =============================================================================

library(themis)
library(rsample)

# -----------------------------------------------------------------------------
# 1. TEMPORAL FEATURES (from date column)
# -----------------------------------------------------------------------------
cat("Creating temporal features...\n")

df <- df %>%
  mutate(
    month           = month(date),
    day_of_week     = wday(date, label = FALSE),
    hour            = hour(date),
    is_weekend      = if_else(wday(date) %in% c(1, 7), 1, 0),
    quarter         = quarter(date),
    season          = case_when(
      month %in% c(8, 9, 10, 11, 12) ~ "early_season",
      month %in% c(1, 2, 3, 4, 5)    ~ "late_season",
      TRUE                            ~ "off_season"
    ),
    is_holiday_season = if_else(month %in% c(12, 1), 1, 0)
  )

cat("Temporal features created: month, day_of_week, hour, is_weekend,",
    "quarter, season, is_holiday_season\n\n")

# -----------------------------------------------------------------------------
# 2. TEAM PERFORMANCE FEATURES (historical aggregations)
# -----------------------------------------------------------------------------
cat("Creating team performance features...\n")

# Sort by date to ensure historical calculation is correct
df <- df %>% arrange(date)

# Home team stats
home_stats <- df %>%
  group_by(homeTeamId) %>%
  mutate(
    home_team_win_rate   = lag(cumsum(outcome == "Home Win") /
                                 row_number(), default = 0),
    home_team_draw_rate  = lag(cumsum(outcome == "Draw") /
                                 row_number(), default = 0),
    home_team_experience = lag(row_number(), default = 0)
  ) %>%
  ungroup() %>%
  select(eventId, home_team_win_rate, home_team_draw_rate, home_team_experience)

# Away team stats
away_stats <- df %>%
  group_by(awayTeamId) %>%
  mutate(
    away_team_win_rate   = lag(cumsum(outcome == "Away Win") /
                                 row_number(), default = 0),
    away_team_draw_rate  = lag(cumsum(outcome == "Draw") /
                                 row_number(), default = 0),
    away_team_experience = lag(row_number(), default = 0)
  ) %>%
  ungroup() %>%
  select(eventId, away_team_win_rate, away_team_draw_rate, away_team_experience)

# Join back to main dataframe
df <- df %>%
  left_join(home_stats, by = "eventId") %>%
  left_join(away_stats, by = "eventId")

cat("Team performance features created\n\n")

# -----------------------------------------------------------------------------
# 3. DERIVED STRENGTH METRICS
# -----------------------------------------------------------------------------
cat("Creating strength differential features...\n")

df <- df %>%
  mutate(
    strength_difference   = home_team_win_rate - away_team_win_rate,
    combined_strength     = home_team_win_rate + away_team_win_rate,
    experience_difference = home_team_experience - away_team_experience
  )

cat("Strength features created: strength_difference, combined_strength,",
    "experience_difference\n\n")

# -----------------------------------------------------------------------------
# 4. LEAGUE FEATURES
# -----------------------------------------------------------------------------
cat("Creating league features...\n")

league_stats <- df %>%
  group_by(leagueId) %>%
  summarise(
    league_home_advantage = mean(outcome == "Home Win", na.rm = TRUE),
    league_draw_rate      = mean(outcome == "Draw",     na.rm = TRUE),
    league_size           = n(),
    .groups = "drop"
  )

df <- df %>% left_join(league_stats, by = "leagueId")

cat("League features created: league_home_advantage, league_draw_rate,",
    "league_size\n\n")

# -----------------------------------------------------------------------------
# 5. VENUE FEATURES
# -----------------------------------------------------------------------------
cat("Creating venue features...\n")

venue_stats <- df %>%
  group_by(venueId) %>%
  summarise(
    venue_home_advantage = mean(outcome == "Home Win", na.rm = TRUE),
    venue_usage          = n(),
    .groups = "drop"
  )

df <- df %>% left_join(venue_stats, by = "venueId")

cat("Venue features created: venue_home_advantage, venue_usage\n\n")

# -----------------------------------------------------------------------------
# 6. HEAD-TO-HEAD FREQUENCY
# -----------------------------------------------------------------------------
cat("Creating head-to-head frequency feature...\n")

h2h_stats <- df %>%
  mutate(matchup_id = paste(pmin(homeTeamId, awayTeamId),
                            pmax(homeTeamId, awayTeamId),
                            sep = "_")) %>%
  group_by(matchup_id) %>%
  summarise(h2h_frequency = n(), .groups = "drop")

df <- df %>%
  mutate(matchup_id = paste(pmin(homeTeamId, awayTeamId),
                            pmax(homeTeamId, awayTeamId),
                            sep = "_")) %>%
  left_join(h2h_stats, by = "matchup_id") %>%
  select(-matchup_id)

cat("Head-to-head feature created: h2h_frequency\n\n")

# -----------------------------------------------------------------------------
# 7. SELECT FINAL FEATURE SET (26 features matching Python)
# -----------------------------------------------------------------------------
cat("Selecting final 26 features...\n")

model_features <- c(
  # Encodings (4)
  "leagueId", "venueId", "homeTeamId", "awayTeamId",
  # Temporal (7)
  "month", "day_of_week", "hour", "is_weekend",
  "quarter", "season", "is_holiday_season",
  # Team performance (6)
  "home_team_win_rate", "away_team_win_rate",
  "home_team_draw_rate", "away_team_draw_rate",
  "home_team_experience", "away_team_experience",
  # Strength metrics (3)
  "strength_difference", "combined_strength", "experience_difference",
  # League features (3)
  "league_home_advantage", "league_draw_rate", "league_size",
  # Venue features (2)
  "venue_home_advantage", "venue_usage",
  # Head-to-head (1)
  "h2h_frequency"
)

# Convert ID columns to factors before modelling
df_model <- df %>%
  select(all_of(model_features), outcome) %>%
  mutate(
    outcome    = factor(outcome,
                        levels = c("Home Win", "Draw", "Away Win")),
    leagueId   = factor(leagueId),
    venueId    = factor(venueId),
    homeTeamId = factor(homeTeamId),
    awayTeamId = factor(awayTeamId),
    season     = factor(season)
  )

# Drop rows with any NA in features
df_model <- df_model %>% drop_na()

cat("Feature count:", length(model_features), "(target: 26)\n")
cat("Rows after dropping NAs:", nrow(df_model), "\n\n")

# -----------------------------------------------------------------------------
# 8. TRAIN / TEST SPLIT (stratified 80/20)
# -----------------------------------------------------------------------------
cat("Creating stratified 80/20 train/test split...\n")

set.seed(42)
data_split <- initial_split(df_model, prop = 0.8, strata = outcome)
train_data <- training(data_split)
test_data  <- testing(data_split)

cat("Training set:", nrow(train_data), "rows\n")
cat("Testing set: ", nrow(test_data),  "rows\n")
cat("Expected ~  : Training ~53,000 / Testing ~13,000\n\n")

# Check class balance in train
cat("Class distribution in training set:\n")
print(table(train_data$outcome))
cat("\n")

# -----------------------------------------------------------------------------
# 9. BUILD TIDYMODELS RECIPE WITH SMOTE
# -----------------------------------------------------------------------------
cat("Building preprocessing recipe with SMOTE...\n")

rec <- recipe(outcome ~ ., data = train_data) %>%
  step_other(leagueId, venueId, homeTeamId, awayTeamId,
             threshold = 0.01) %>%
  step_dummy(leagueId, venueId, homeTeamId, awayTeamId, season) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(outcome, over_ratio = 1)

cat("Recipe created with steps:\n")
cat("  - step_other()     : handle high-cardinality IDs\n")
cat("  - step_dummy()     : encode categoricals\n")
cat("  - step_normalize() : StandardScaler equivalent\n")
cat("  - step_smote()     : SMOTE class balancing\n\n")

# Prep the recipe to verify it works
cat("Prepping recipe (applying SMOTE)...\n")
rec_prepped <- prep(rec, training = train_data)
train_baked <- bake(rec_prepped, new_data = NULL)

cat("After SMOTE — class distribution:\n")
print(table(train_baked$outcome))
cat("\n")

# -----------------------------------------------------------------------------
# 10. SAVE PROCESSED DATA
# -----------------------------------------------------------------------------
cat("Saving processed datasets...\n")

saveRDS(rec_prepped,  here("classification", "output", "recipe_prepped.rds"))
saveRDS(train_data,   here("classification", "output", "train_data.rds"))
saveRDS(test_data,    here("classification", "output", "test_data.rds"))
saveRDS(train_baked,  here("classification", "output", "train_baked.rds"))

cat("Saved: recipe_prepped.rds\n")
cat("Saved: train_data.rds\n")
cat("Saved: test_data.rds\n")
cat("Saved: train_baked.rds\n")

cat("\n✓ Issue #6 complete — feature engineering done,",
    "processed data saved to classification/output/\n")




# =============================================================================
# Football Match Outcome Prediction — Classification
# Group 8 | Reproducible Research | University of Warsaw 2025-2026
# =============================================================================
# Author  : Ramik Sharma (477656)
# Issue   : #7 — Model Training & Evaluation
# Dataset : ESPN Soccer Data 2024-2026 (Kaggle)
# =============================================================================
# Picks up from the RDS files saved by Issue #6.
# Trains RF, GB, XGBoost, NN, and Stacking Ensemble via tidymodels.
# Writes comprehensive_results.txt to classification/output/
# =============================================================================
library(tidymodels)
library(ranger)
library(xgboost)
library(brulee)
library(stacks)
library(yardstick)

options(xgboost.verbose = 0)
set.seed(42)

output_dir     <- here("classification", "output")
outcome_levels <- c("Home Win", "Draw", "Away Win")
# Re-enforcing factor levels on the data that Issue #6 already created (they are already in the environment
# ; this is a safety re-casting)
train_data <- train_data %>%
  mutate(outcome = factor(outcome, levels = outcome_levels))
test_data  <- test_data  %>%
  mutate(outcome = factor(outcome, levels = outcome_levels))

cat("Training rows:", nrow(train_data), "| Testing rows:", nrow(test_data), "\n\n")

# =============================================================================
# RECIPE — rebuilt so every CV fold bakes its own copy
# =============================================================================

id_cols <- c("leagueId", "venueId", "homeTeamId", "awayTeamId")

base_recipe <- recipe(outcome ~ ., data = train_data) %>%
  step_other(all_of(id_cols), threshold = 0.01) %>%
  step_dummy(all_of(id_cols), season, one_hot = FALSE) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(outcome, over_ratio = 1)

# =============================================================================
# BALANCED CLASS WEIGHTS for Random Forest
# Mirrors sklearn class_weight='balanced': w_j = n / (k * n_j)
# =============================================================================
class_counts  <- table(train_data$outcome)
balanced_wts  <- sum(class_counts) / (length(class_counts) * as.numeric(class_counts))
names(balanced_wts) <- names(class_counts)
cat("Balanced class weights:\n"); print(round(balanced_wts, 4)); cat("\n")

# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================
# Random Forest — Python: n_estimators=300, max_depth=12, min_samples_split=5
rf_spec <- rand_forest(trees = 300, min_n = 5) %>%
  set_engine("ranger",
             max.depth     = 12,
             class.weights = balanced_wts,
             importance    = "impurity",
             seed          = 42,
             num.threads   = 1) %>%
  set_mode("classification")

# Gradient Boosting — Python: n_estimators=300, max_depth=8, lr=0.1, subsample=0.8
gb_spec <- boost_tree(trees = 300, tree_depth = 8,
                      learn_rate = 0.1, sample_size = 0.8) %>%
  set_engine("xgboost", booster = "gbtree",
             eval_metric = "mlogloss", nthread = 1, seed = 42) %>%
  set_mode("classification")

# XGBoost — Python: n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, colsample_bytree=0.8
xgb_spec <- boost_tree(trees = 300, tree_depth = 6,
                       learn_rate = 0.1, sample_size = 0.8, mtry = 0.8) %>%
  set_engine("xgboost", counts = FALSE,
             eval_metric = "mlogloss", nthread = 1, seed = 42) %>%
  set_mode("classification")

# Neural Network — Python: hidden=(150,100,50), alpha=0.001, lr_init=0.001, max_iter=300
nn_spec <- mlp(hidden_units = c(150L, 100L, 50L),
               epochs = 300, learn_rate = 0.001, dropout = 0.001) %>%
  set_engine("brulee", seed = 42) %>%
  set_mode("classification")

cat("Model specs defined for all models: RF | GB | XGBoost | NN\n\n")

# =============================================================================
# ADDING WORKFLOWS
# =============================================================================
rf_wf  <- workflow() %>% add_recipe(base_recipe) %>% add_model(rf_spec)
gb_wf  <- workflow() %>% add_recipe(base_recipe) %>% add_model(gb_spec)
xgb_wf <- workflow() %>% add_recipe(base_recipe) %>% add_model(xgb_spec)
nn_wf  <- workflow() %>% add_recipe(base_recipe) %>% add_model(nn_spec)

# =============================================================================
# 5-FOLD STRATIFIED CV
# Mirrors Python: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# =============================================================================
set.seed(42)
folds <- vfold_cv(train_data, v = 5, strata = outcome)

cv_metrics  <- metric_set(accuracy, f_meas, roc_auc)
ctrl_stack  <- control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = FALSE)
ctrl_plain  <- control_resamples(save_pred = TRUE, verbose = FALSE)

cat("Fitting CV — Random Forest...\n");       set.seed(42); rf_res  <- fit_resamples(rf_wf,  folds, metrics = cv_metrics, control = ctrl_stack)
cat("Fitting CV — Gradient Boosting...\n");   set.seed(42); gb_res  <- fit_resamples(gb_wf,  folds, metrics = cv_metrics, control = ctrl_stack)
cat("Fitting CV — XGBoost...\n");             set.seed(42); xgb_res <- fit_resamples(xgb_wf, folds, metrics = cv_metrics, control = ctrl_stack)
cat("Fitting CV — Neural Network...\n");      set.seed(42); nn_res  <- fit_resamples(nn_wf,  folds, metrics = cv_metrics, control = ctrl_plain)
cat("All CV fits - complete.\n\n")

cv_summary <- bind_rows(
  collect_metrics(rf_res)  %>% mutate(model = "Random Forest"),
  collect_metrics(gb_res)  %>% mutate(model = "Gradient Boosting"),
  collect_metrics(xgb_res) %>% mutate(model = "XGBoost"),
  collect_metrics(nn_res)  %>% mutate(model = "Neural Network")
) %>% select(model, .metric, mean, std_err)

cat("5-Fold CV Mean Metrics:\n"); print(cv_summary, n = Inf); cat("\n")

# =============================================================================
# STACKING ENSEMBLE
# RF + GB + XGB as base learners; elastic-net meta-learner (= LR equivalent)
# =============================================================================
cat("Building stacking ensemble.......\n")
set.seed(42)
stack_model <- stacks() %>%
  add_candidates(rf_res)  %>%
  add_candidates(gb_res)  %>%
  add_candidates(xgb_res) %>%
  blend_predictions(penalty = 10 ^ seq(-6, -1, length.out = 20), mixture = 1) %>%
  fit_members()
cat("Stacking ensemble is now ready.\n\n")

# =============================================================================
# FITTING FINAL MODELS ON FULL TRAINING DATA (for test-set predictions)
# =============================================================================
cat("Fitting final models on full training data.....\n")
set.seed(42); rf_final  <- fit(rf_wf,  train_data)
set.seed(42); gb_final  <- fit(gb_wf,  train_data)
set.seed(42); xgb_final <- fit(xgb_wf, train_data)
set.seed(42); nn_final  <- fit(nn_wf,  train_data)
cat("All models fitted now.\n\n")

# =============================================================================
# EVALUATING TEST-SET
# =============================================================================
get_test_preds <- function(fitted_wf, new_data) {
  bind_cols(
    new_data %>% select(outcome),
    predict(fitted_wf, new_data, type = "class"),
    predict(fitted_wf, new_data, type = "prob")
  )
}

rf_preds    <- get_test_preds(rf_final,  test_data)
gb_preds    <- get_test_preds(gb_final,  test_data)
xgb_preds   <- get_test_preds(xgb_final, test_data)
nn_preds    <- get_test_preds(nn_final,  test_data)
stack_preds <- bind_cols(
  test_data %>% select(outcome),
  predict(stack_model, test_data, type = "class"),
  predict(stack_model, test_data, type = "prob")
)
all_preds <- list(
  "Random Forest"     = rf_preds,
  "Gradient Boosting" = gb_preds,
  "XGBoost"           = xgb_preds,
  "Neural Network"    = nn_preds,
  "Stacking Ensemble" = stack_preds
)

prob_cols <- paste0(".pred_", outcome_levels)

eval_results <- map_dfr(names(all_preds), function(nm) {
  p   <- all_preds[[nm]]
  acc <- accuracy(p, truth = outcome, estimate = .pred_class)$.estimate
  f1  <- f_meas(p,   truth = outcome, estimate = .pred_class, estimator = "weighted")$.estimate
  auc <- roc_auc(p,  truth = outcome, any_of(prob_cols), estimator = "macro_weighted")$.estimate
  tibble(model = nm, accuracy = acc, f1_weighted = f1, auc = auc)
})

cat(sprintf("%-20s  %8s  %8s  %8s\n", "Model", "Accuracy", "F1(wt)", "AUC"))
cat(strrep("-", 52), "\n")
for (i in seq_len(nrow(eval_results))) {
  r <- eval_results[i,]
  cat(sprintf("%-20s  %7.2f%%  %8.4f  %8.4f\n", r$model, r$accuracy*100, r$f1_weighted, r$auc))
}
cat("\n")

best_name  <- eval_results$model[which.max(eval_results$accuracy)]
best_preds <- all_preds[[best_name]]
cat("Best model:", best_name, "\n\n")

cm_best <- conf_mat(best_preds, truth = outcome, estimate = .pred_class)
print(cm_best)

# Per-class precision / recall / F1 from confusion matrix
compute_per_class <- function(cm_tbl, classes) {
  map_dfr(classes, function(cls) {
    tp   <- cm_tbl[cls, cls]
    fp   <- sum(cm_tbl[, cls]) - tp
    fn   <- sum(cm_tbl[cls, ]) - tp
    prec <- if ((tp+fp)>0) tp/(tp+fp) else 0
    rec  <- if ((tp+fn)>0) tp/(tp+fn) else 0
    f1   <- if ((prec+rec)>0) 2*prec*rec/(prec+rec) else 0
    tibble(class = cls, precision = prec, recall = rec, f1 = f1)
  })
}

per_class     <- compute_per_class(cm_best$table, outcome_levels)
all_per_class <- map_dfr(names(all_preds), function(nm) {
  cm_i <- conf_mat(all_preds[[nm]], truth = outcome, estimate = .pred_class)
  compute_per_class(cm_i$table, outcome_levels) %>% mutate(model = nm)
})

cat("\nPer-class metrics for the", best_name, ":\n")
print(per_class)

# =============================================================================
# SAVING ARTEFACTS FOR ISSUE #8
# =============================================================================
saveRDS(rf_final,      file.path(output_dir, "rf_final.rds"))
saveRDS(gb_final,      file.path(output_dir, "gb_final.rds"))
saveRDS(xgb_final,     file.path(output_dir, "xgb_final.rds"))
saveRDS(nn_final,      file.path(output_dir, "nn_final.rds"))
saveRDS(stack_model,   file.path(output_dir, "stack_model.rds"))
saveRDS(eval_results,  file.path(output_dir, "eval_results.rds"))
saveRDS(all_preds,     file.path(output_dir, "all_preds.rds"))
saveRDS(all_per_class, file.path(output_dir, "all_per_class.rds"))
saveRDS(per_class,     file.path(output_dir, "per_class_best.rds"))
saveRDS(cm_best,       file.path(output_dir, "cm_best.rds"))

# Write comprehensive_results.txt
best_row <- eval_results %>% filter(model == best_name)
lines <- c(
  "FOOTBALL MATCH OUTCOME CLASSIFICATION PROJECT",
  strrep("=", 60), "",
  "BEST MODEL PERFORMANCE:", strrep("-", 25),
  sprintf("Algorithm: %s", best_name),
  sprintf("Accuracy: %.4f (%.2f%%)", best_row$accuracy, best_row$accuracy*100),
  sprintf("F1-Score: %.4f",   best_row$f1_weighted),
  sprintf("AUC Score: %.4f",  best_row$auc), "",
  "ALL MODEL RESULTS:", strrep("-", 20)
)
for (i in seq_len(nrow(eval_results))) {
  r <- eval_results[i,]
  lines <- c(lines,
             sprintf("%s:", r$model),
             sprintf("  Accuracy: %.4f", r$accuracy),
             sprintf("  F1-Score: %.4f", r$f1_weighted),
             sprintf("  AUC Score: %.4f", r$auc), "")
}
lines <- c(lines,
           "PYTHON BASELINE COMPARISON:", strrep("-", 30),
           sprintf("  %-20s  %8s  %8s  %8s", "Model","Accuracy","F1","AUC"),
           sprintf("  %-20s  %8s  %8s  %8s", "Random Forest (Py)",    "62.87%","0.6267","0.8176"),
           sprintf("  %-20s  %8s  %8s  %8s", "Gradient Boost (Py)",   "64.16%","0.6412","0.8265"),
           sprintf("  %-20s  %8s  %8s  %8s", "XGBoost (Py)",          "64.84%","0.6476","0.8326"),
           sprintf("  %-20s  %8s  %8s  %8s", "Neural Network (Py)",   "59.45%","0.5941","0.7758"),
           sprintf("  %-20s  %8s  %8s  %8s", "Stacking (Py)",         "65.36%","0.6533","0.8347")
)
writeLines(lines, file.path(output_dir, "comprehensive_results.txt"))

cat("\n✓ Issue # 7 - Complete; models trained, comprehensive_results.txt saved\n")


# =============================================================================
# Football Match Outcome Prediction — Classification
# Group 8 | Reproducible Research | University of Warsaw 2025-2026
# =============================================================================
# Author  : Ramik Sharma (477656)
# Issue   : #8 — Visualisations, Error Analysis & Ethical Considerations
# Dataset : ESPN Soccer Data 2024-2026 (Kaggle)
# =============================================================================
# Loads artefacts from Issue #7 and produces three plots, error_analysis.txt,
# and ethical_considerations.txt — all to classification/output/
# =============================================================================

library(patchwork)

cat("\n", strrep("=", 60), "\n", sep = "")
cat("ISSUE # 8. — VISUALISATIONS, ERROR ANALYSIS AND ETHICS\n")
cat(strrep("=", 60), "\n\n", sep = "")

# The objects eval_results, all_preds, all_per_class, cm_best, best_name,
# per_class, outcome_levels, output_dir are all already in the environment
# from Issue # 7. If running this section standalone, ensure reloading them:
if (!exists("eval_results")) {
  eval_results  <- readRDS(file.path(output_dir, "eval_results.rds"))
  all_preds     <- readRDS(file.path(output_dir, "all_preds.rds"))
  all_per_class <- readRDS(file.path(output_dir, "all_per_class.rds"))
  cm_best       <- readRDS(file.path(output_dir, "cm_best.rds"))
  per_class     <- readRDS(file.path(output_dir, "per_class_best.rds"))
  rf_final      <- readRDS(file.path(output_dir, "rf_final.rds"))
  best_name     <- eval_results$model[which.max(eval_results$accuracy)]
}

model_colours <- c(
  "Random Forest"     = "#4E79A7",
  "Gradient Boosting" = "#F28E2B",
  "XGBoost"           = "#E15759",
  "Neural Network"    = "#76B7B2",
  "Stacking Ensemble" = "#59A14F"
)

bar_theme <- theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
        panel.grid.major.x = element_blank(),
        legend.position = "none")

# =============================================================================
# PLOT 1 — Dashboard - Model performance (with a 2x2 grid)
# =============================================================================
p_acc <- ggplot(eval_results,
                aes(x = reorder(model, accuracy), y = accuracy, fill = model)) +
  geom_col(alpha = 0.85, width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", accuracy)), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = model_colours) +
  scale_y_continuous(limits = c(0,1), expand = expansion(mult = c(0, 0.07))) +
  labs(title = "Accuracy Comparison", x = NULL, y = "Accuracy") +
  bar_theme

p_f1 <- ggplot(eval_results,
               aes(x = reorder(model, f1_weighted), y = f1_weighted, fill = model)) +
  geom_col(alpha = 0.85, width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", f1_weighted)), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = model_colours) +
  scale_y_continuous(limits = c(0,1), expand = expansion(mult = c(0, 0.07))) +
  labs(title = "F1-Score Comparison", x = NULL, y = "F1-Score (Weighted)") +
  bar_theme

p_auc <- ggplot(eval_results,
                aes(x = reorder(model, auc), y = auc, fill = model)) +
  geom_col(alpha = 0.85, width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", auc)), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = model_colours) +
  scale_y_continuous(limits = c(0,1), expand = expansion(mult = c(0, 0.07))) +
  labs(title = "AUC Score Comparison", x = NULL, y = "AUC Score") +
  bar_theme

cm_tidy <- as_tibble(cm_best$table) %>%
  rename(Truth = truth, Prediction = estimate, Count = n) %>%
  mutate(Truth      = factor(Truth,      levels = outcome_levels),
         Prediction = factor(Prediction, levels = outcome_levels))

p_cm <- ggplot(cm_tidy, aes(x = Prediction, y = Truth, fill = Count)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = Count), size = 4.5, fontface = "bold") +
  scale_fill_gradient(low = "#DEEBF7", high = "#2171B5") +
  scale_x_discrete(limits = outcome_levels) +
  scale_y_discrete(limits = rev(outcome_levels)) +
  labs(title = sprintf("Confusion Matrix — %s", best_name),
       x = "Predicted", y = "Actual", fill = "Count") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

combined_plot <- (p_acc | p_f1) / (p_auc | p_cm) +
  plot_annotation(
    title    = "Football Match Outcome — Model Performance Dashboard",
    subtitle = "Accuracy, F1, AUC and best-model confusion matrix",
    theme    = theme(plot.title    = element_text(size = 14, face = "bold"),
                     plot.subtitle = element_text(size = 10, colour = "grey40"))
  )

ggsave(file.path(output_dir, "model_performance_comparison.png"),
       combined_plot, width = 16, height = 12, dpi = 300, bg = "white")
cat("Saved: model_performance_comparison.png\n")

# =============================================================================
# PLOT 2 — Per-class precision and recall (grouping based bar charts)
# =============================================================================
all_per_class <- all_per_class %>%
  mutate(class = factor(class, levels = outcome_levels),
         model = factor(model, levels = c("Random Forest","Gradient Boosting",
                                          "XGBoost","Neural Network","Stacking Ensemble")))

p_prec <- ggplot(all_per_class,
                 aes(x = class, y = precision, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75, alpha = 0.85) +
  scale_fill_manual(values = model_colours, name = "Model") +
  scale_y_continuous(limits = c(0,1), expand = expansion(mult = c(0,0.05))) +
  labs(title = "Precision by Class and Model", x = "Outcome Class", y = "Precision") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())

p_rec <- ggplot(all_per_class,
                aes(x = class, y = recall, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75, alpha = 0.85) +
  scale_fill_manual(values = model_colours, name = "Model") +
  scale_y_continuous(limits = c(0,1), expand = expansion(mult = c(0,0.05))) +
  labs(title = "Recall by Class and Model", x = "Outcome Class", y = "Recall") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())

per_class_plot <- (p_prec | p_rec) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "per_class_performance.png"),
       per_class_plot, width = 16, height = 6, dpi = 300, bg = "white")
cat("Saved: per_class_performance.png\n")

# =============================================================================
# PLOT 3 — Top 15 feature importance from the Random Forest
# =============================================================================
rf_engine <- extract_fit_engine(rf_final)

if (!is.null(rf_engine$variable.importance) &&
    length(rf_engine$variable.importance) > 0) {
  importance_df <- tibble(
    feature    = names(rf_engine$variable.importance),
    importance = as.numeric(rf_engine$variable.importance)
  ) %>%
    arrange(desc(importance)) %>%
    slice_head(n = 15)
  
  p_imp <- ggplot(importance_df,
                  aes(x = reorder(feature, importance), y = importance)) +
    geom_col(fill = "#4E79A7", alpha = 0.85, width = 0.75) +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(title    = "Top 15 Feature Importances (Random Forest)",
         subtitle = "Gini impurity-based importance scores",
         x = "Feature", y = "Importance Score") +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 9))
  
  ggsave(file.path(output_dir, "feature_importance.png"),
         p_imp, width = 12, height = 8, dpi = 300, bg = "white")
  cat("Saved: feature_importance.png\n\n")
} else {
  cat("WARNING: importance not available — ensure importance='impurity' in rf_spec\n\n")
}

# =============================================================================
# ERROR ANALYSIS FOR MODEL
# =============================================================================
best_preds_ea <- all_preds[[best_name]] %>%
  mutate(outcome     = factor(outcome,     levels = outcome_levels),
         .pred_class = factor(.pred_class, levels = outcome_levels),
         misclassified = outcome != .pred_class)

n_test    <- nrow(best_preds_ea)
n_errors  <- sum(best_preds_ea$misclassified)
error_pct <- n_errors / n_test * 100

per_class_err <- map_dfr(outcome_levels, function(cls) {
  rows  <- best_preds_ea %>% filter(outcome == cls)
  errs  <- rows %>% filter(misclassified)
  wrong <- if (nrow(errs) > 0)
    as.character(errs %>% count(.pred_class, sort=TRUE) %>% slice(1) %>% pull(.pred_class))
  else "N/A"
  tibble(class = cls, total = nrow(rows), errors = nrow(errs),
         error_rate_pct = nrow(errs)/nrow(rows)*100, most_common_wrong = wrong)
})

err_lines <- c(
  "ERROR ANALYSIS REPORT", strrep("=", 50), "",
  sprintf("Best Model: %s", best_name),
  sprintf("Total Errors: %d / %d (%.1f%%)", n_errors, n_test, error_pct), ""
)
for (i in seq_len(nrow(per_class_err))) {
  r <- per_class_err[i,]
  err_lines <- c(err_lines,
                 sprintf("%s:", r$class),
                 sprintf("  Total samples : %d", r$total),
                 sprintf("  Misclassified : %d", r$errors),
                 sprintf("  Error rate    : %.1f%%", r$error_rate_pct),
                 sprintf("  Most often misclassified as: %s", r$most_common_wrong), "")
}
err_lines <- c(err_lines,
               "PYTHON BASELINE (Stacking Ensemble):",
               "  Total Errors : 5516 / 15925 (34.6%)",
               "  Home Win     : 33.5% error rate",
               "  Draw         : 40.3% error rate",
               "  Away Win     : 30.2% error rate")
writeLines(err_lines, file.path(output_dir, "error_analysis.txt"))
cat("Saved: error_analysis.txt\n")

# =============================================================================
# ETHICAL CONSIDERATIONS AND BIAS ANALYSIS
# =============================================================================
eth_lines <- c(
  "ETHICAL CONSIDERATIONS AND BIAS ANALYSIS",
  strrep("=", 45), "",
  "Project: Football Match Outcome Classification (R Replication)",
  "Group 8 | Reproducible Research | University of Warsaw 2025-2026", "",
  "1. DATA BIAS CONSIDERATIONS:",
  "   - Historical bias: model learns from past records which may encode",
  "     structural advantages of well-funded or dominant leagues.",
  "   - Temporal bias: team form changes; older seasons may not reflect",
  "     present-day performance.",
  "   - Geographic bias: European top-flight leagues dominate the dataset,",
  "     leaving lower-profile and non-European football under-represented.", "",
  "2. FAIRNESS CONCERNS:",
  "   - Team fairness: model may over-predict wins for historically strong teams.",
  "   - League fairness: generalisation across leagues is uneven.",
  "   - Venue fairness: home advantage signal may amplify structural inequality.", "",
  "3. POTENTIAL MISUSE:",
  "   - Gambling: must NOT be used to inform betting decisions.",
  "     Error rate is ~34-35%; over-reliance could cause financial harm.",
  "   - Match fixing: public predictions could motivate manipulation.",
  "   - Financial decisions: transfers or sponsorship must not rely on this model.", "",
  "4. TRANSPARENCY:",
  "   - Feature importance charts provided (Random Forest Gini scores).",
  "   - Model limitations documented; ~65% accuracy is a realistic ceiling.",
  "   - Confidence calibration not formally evaluated.",
  "   - All code is open, version-controlled, reproducible via renv.lock.", "",
  "5. RECOMMENDATIONS:",
  "   - Retrain at the start of each season with recent data.",
  "   - Monitor prediction distributions per league for demographic drift.",
  "   - Provide confidence intervals alongside point predictions.",
  "   - Include responsible-use disclaimer before any deployment.",
  "   - Conduct a bias audit across leagues and competitive tiers.", "",
  strrep("-", 45),
  "Prepared for Reproducible Research course, University of Warsaw 2025-2026)",
  "Replicates ethical section from the original Python ML2 project",
  "(Rajat Dogra & Umair Aziz, January 2026)."
)
writeLines(eth_lines, file.path(output_dir, "ethical_considerations.txt"))
cat("Saved: ethical_considerations.txt\n")

cat("\n✓ Issue # 8 - Completed; plots and text reports saved to classification/output/\n")
