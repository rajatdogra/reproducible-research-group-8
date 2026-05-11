# =============================================================================
# Issue #4: [Regression] Hyperparameter tuning, ensembles & final evaluation
# Replicates Sections 5 (Hyperparameter Optimization) and 6 (Ensemble Methods)
# of the Python ML2 notebook in R using tidymodels + stacks.
#
# Prerequisites: run 03_baseline_models.R first (creates split.rds and
# baseline_ranking.csv that this script consumes).
#
# Encoding strategy: step_integer() matches the Python pipeline's LabelEncoder.
# Target column and feature selection verified against:
#   regression/original_python/Cricket_Score_Prediction_Final_Submission.ipynb
# =============================================================================

library(tidyverse)
library(tidymodels)
library(stacks)
library(ranger)
library(xgboost)
library(yardstick)
library(ggrepel)
library(here)

set.seed(123)
tidymodels_prefer()

out_dir <- here("regression", "output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1. Reuse split from issue #3 ------------------------------------------
split_obj <- readRDS(file.path(out_dir, "split.rds"))
train <- split_obj$train
test  <- split_obj$test
folds <- split_obj$folds

# Target confirmed from Python notebook: y = df['final_score']
target_col <- "final_score"

# Recipe matches issue #3 (same encoding, same dropped columns)
base_rec <- recipe(as.formula(paste(target_col, "~ .")), data = train) %>%
  step_rm(match_id, season) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_integer(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

metric_set_reg <- metric_set(rsq, mae, rmse)
# control_stack_grid() under the hood — we add verbose for visible progress
ctrl_grid <- control_grid(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)

# ============================================================================
# 2. GRADIENT BOOSTING TUNING (81 configs)
# ============================================================================
gbm_spec <- boost_tree(
  trees       = tune(),
  learn_rate  = tune(),
  tree_depth  = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

gbm_wf <- workflow(base_rec, gbm_spec)

gbm_grid <- tidyr::crossing(
  trees       = c(150L, 200L, 300L),
  learn_rate  = c(0.05, 0.10, 0.15),
  tree_depth  = c(4L, 6L, 8L),
  sample_size = c(0.8, 0.9, 1.0)
)

cat(">> Tuning XGBoost over", nrow(gbm_grid), "configs x 5 folds...\n")
gbm_tuned <- tune_grid(
  gbm_wf,
  resamples = folds,
  grid      = gbm_grid,
  metrics   = metric_set_reg,
  control   = ctrl_grid
)
saveRDS(gbm_tuned, file.path(out_dir, "gbm_tuned.rds"))  # checkpoint
gbm_best <- select_best(gbm_tuned, metric = "rsq")

# ============================================================================
# 3. RANDOM FOREST TUNING (9 configs)
# Note: parsnip's rand_forest() does not expose ranger's max.depth as a
# tunable argument — it lives in set_engine() and can't be tuned via the
# standard grid path without extra parameter-object plumbing. We tune trees
# and min_n only; depth defaults to unlimited (ranger's default).
# ============================================================================
rf_spec <- rand_forest(
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", num.threads = max(1, parallel::detectCores() - 1)) %>%
  set_mode("regression")

rf_wf <- workflow(base_rec, rf_spec)

rf_grid <- tidyr::crossing(
  trees = c(150L, 200L, 300L),
  min_n = c(2L, 5L, 10L)
)

cat(">> Tuning Random Forest over", nrow(rf_grid), "configs x 5 folds...\n")
rf_tuned <- tune_grid(
  rf_wf,
  resamples = folds,
  grid      = rf_grid,
  metrics   = metric_set_reg,
  control   = ctrl_grid
)
saveRDS(rf_tuned, file.path(out_dir, "rf_tuned.rds"))  # checkpoint
rf_best <- select_best(rf_tuned, metric = "rsq")

# ============================================================================
# 4. EXTRA TREES (ranger splitrule = "extratrees")
# ============================================================================
et_spec <- rand_forest(trees = 200, min_n = 5) %>%
  set_engine("ranger",
             splitrule  = "extratrees",
             num.threads = max(1, parallel::detectCores() - 1)) %>%
  set_mode("regression")

et_wf <- workflow(base_rec, et_spec)

# ============================================================================
# 5. VOTING ENSEMBLE via stacks (blends top candidates from GBM + RF tunings)
# ============================================================================
cat(">> Building stacked ensemble...\n")
model_stack <- stacks() %>%
  add_candidates(gbm_tuned) %>%
  add_candidates(rf_tuned) %>%
  blend_predictions() %>%
  fit_members()

# ============================================================================
# 6. FINAL TEST-SET EVALUATION
# ============================================================================
eval_on_test <- function(wf, best_params = NULL, name) {
  final_wf  <- if (is.null(best_params)) wf else finalize_workflow(wf, best_params)
  final_fit <- fit(final_wf, data = train)
  preds <- predict(final_fit, new_data = test) %>%
    bind_cols(test %>% select(all_of(target_col))) %>%
    rename(truth = !!sym(target_col), estimate = .pred)
  metrics <- metric_set_reg(preds, truth = truth, estimate = estimate) %>%
    mutate(Model = name)
  list(metrics = metrics, preds = preds)
}

gbm_eval <- eval_on_test(gbm_wf, gbm_best, "XGBoost (Optimised)")
rf_eval  <- eval_on_test(rf_wf,  rf_best,  "Random Forest (Optimised)")
et_eval  <- eval_on_test(et_wf,  NULL,     "Extra Trees")

stack_preds <- predict(model_stack, new_data = test) %>%
  bind_cols(test %>% select(all_of(target_col))) %>%
  rename(truth = !!sym(target_col), estimate = .pred)
stack_metrics <- metric_set_reg(stack_preds, truth = truth, estimate = estimate) %>%
  mutate(Model = "Voting Ensemble (Stack)")

optimised_metrics <- bind_rows(
  gbm_eval$metrics, rf_eval$metrics, et_eval$metrics, stack_metrics
)

# ---- Load baseline ranking from issue #3 and combine -----------------------
baseline_path <- file.path(out_dir, "baseline_ranking.csv")
baseline_long <- readr::read_csv(baseline_path, show_col_types = FALSE) %>%
  select(Model, R2, MAE, RMSE) %>%
  pivot_longer(c(R2, MAE, RMSE), names_to = ".metric", values_to = ".estimate") %>%
  mutate(.metric = recode(.metric, R2 = "rsq", MAE = "mae", RMSE = "rmse"))

all_metrics <- bind_rows(
  baseline_long,
  optimised_metrics %>% select(Model, .metric, .estimate)
)

final_ranking <- all_metrics %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(rsq)) %>%
  mutate(Rank = row_number(), .before = 1) %>%
  rename(R2 = rsq, MAE = mae, RMSE = rmse)

cat("\n=== FINAL RANKING ===\n")
print(final_ranking)

# ---- Optimisation impact table ---------------------------------------------
get_r2 <- function(model_name) {
  v <- final_ranking %>% filter(Model == model_name) %>% pull(R2)
  if (length(v) == 0) NA_real_ else v[1]
}

impact_table <- tibble(
  Model        = c("XGBoost", "Random Forest"),
  Baseline_R2  = c(get_r2("XGBoost"),             get_r2("Random Forest")),
  Optimised_R2 = c(get_r2("XGBoost (Optimised)"), get_r2("Random Forest (Optimised)"))
) %>%
  mutate(Improvement = Optimised_R2 - Baseline_R2)

cat("\n=== OPTIMISATION IMPACT ===\n")
print(impact_table)

# ============================================================================
# 7. ggplot2 VISUALISATION DASHBOARD
# ============================================================================
p1 <- final_ranking %>%
  ggplot(aes(x = reorder(Model, R2), y = R2, fill = R2)) +
  geom_col() +
  coord_flip() +
  scale_fill_viridis_c() +
  labs(title = "Model Performance Ranking (R^2)", x = NULL, y = "R^2") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

p2 <- final_ranking %>%
  ggplot(aes(x = MAE, y = RMSE, label = Model)) +
  geom_point(size = 3, colour = "steelblue") +
  geom_text_repel(size = 3, max.overlaps = Inf) +
  labs(title = "MAE vs RMSE across models") +
  theme_minimal(base_size = 11)

best_name <- final_ranking$Model[1]
best_preds <- list(
  "XGBoost (Optimised)"       = gbm_eval$preds,
  "Random Forest (Optimised)" = rf_eval$preds,
  "Extra Trees"               = et_eval$preds,
  "Voting Ensemble (Stack)"   = stack_preds
)[[best_name]]

p3 <- best_preds %>%
  ggplot(aes(x = truth, y = estimate)) +
  geom_point(alpha = 0.3, colour = "steelblue") +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed") +
  labs(title = paste("Predicted vs Actual -", best_name),
       x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 11)

ggsave(file.path(out_dir, "ranking.png"),             p1, width = 8, height = 5, dpi = 150)
ggsave(file.path(out_dir, "mae_vs_rmse.png"),         p2, width = 7, height = 5, dpi = 150)
ggsave(file.path(out_dir, "predicted_vs_actual.png"), p3, width = 7, height = 5, dpi = 150)

readr::write_csv(final_ranking, file.path(out_dir, "final_ranking.csv"))
readr::write_csv(impact_table,  file.path(out_dir, "optimization_impact.csv"))
saveRDS(model_stack,            file.path(out_dir, "model_stack.rds"))

cat("\nTuning + ensembles complete. All outputs saved to", out_dir, "\n")
