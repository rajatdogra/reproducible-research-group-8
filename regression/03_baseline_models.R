# =============================================================================
# Issue #3: [Regression] Baseline model training
# Replicates Section 4 (Model Training) of the Python ML2 notebook in R.
# Trains 6 baseline regression models on IPL cricket score data with 5-fold CV.
# Output: regression/output/baseline_ranking.csv
#
# Encoding strategy: step_integer() matches the Python pipeline's LabelEncoder.
# Target column and feature selection verified against:
#   regression/original_python/Cricket_Score_Prediction_Final_Submission.ipynb
# =============================================================================

library(tidyverse)
library(tidymodels)
library(glmnet)
library(ranger)
library(xgboost)
library(nnet)        # neural net engine (alternative: brulee)
library(yardstick)
library(here)

set.seed(123)
tidymodels_prefer()

# ---- 1. Load data ----------------------------------------------------------
data_path <- here("regression", "data", "regression_dataset.csv")
df <- readr::read_csv(data_path, show_col_types = FALSE)

# Target confirmed from Python notebook: y = df['final_score']
target_col <- "final_score"
stopifnot(target_col %in% names(df))

df <- df %>%
  filter(!is.na(.data[[target_col]])) %>%
  mutate(target_score = tidyr::replace_na(target_score, 0)) %>%   # match Python NA fill
  mutate(across(where(is.character), as.factor))

# ---- 2. Train/test split + 5-fold CV ---------------------------------------
split <- initial_split(df, prop = 0.8)
train <- training(split)
test  <- testing(split)

folds <- vfold_cv(train, v = 5)

# Persist the split so issue #4 reuses the same train/test
out_dir <- here("regression", "output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(list(split = split, train = train, test = test, folds = folds),
        file.path(out_dir, "split.rds"))

# ---- 3. Recipes ------------------------------------------------------------
# - step_rm: drop identifier (match_id) and redundant column (season ~ year)
# - step_unknown: assign "unknown" level to unseen factor levels at predict time
# - step_integer: label-encode factors (matches Python's LabelEncoder)
# - step_zv: drop zero-variance predictors
# - step_normalize: scale numerics (helps glmnet; tree models are scale-invariant)
base_rec <- recipe(as.formula(paste(target_col, "~ .")), data = train) %>%
  step_rm(match_id, season) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_integer(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

poly_rec <- base_rec %>%
  # poly() requires degree < n_unique, so exclude binary cols (innings, toss_decision)
  step_poly(wickets_lost, balls_faced, year, month, day, target_score, degree = 2)

# ---- 4. Model specs --------------------------------------------------------
lr_spec <- linear_reg() %>%
  set_engine("lm")

ridge_spec <- linear_reg(penalty = 0.01, mixture = 0) %>%
  set_engine("glmnet")

# Polynomial Ridge shares the spec - only the recipe differs
rf_spec <- rand_forest(trees = 100) %>%
  set_engine("ranger", num.threads = max(1, parallel::detectCores() - 1)) %>%
  set_mode("regression")

xgb_spec <- boost_tree(trees = 100) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

nn_spec <- mlp(hidden_units = 16, epochs = 100) %>%
  set_engine("nnet", trace = FALSE, MaxNWts = 5000) %>%
  set_mode("regression")

# ---- 5. Workflows ----------------------------------------------------------
workflows_list <- list(
  "Linear Regression" = workflow(base_rec, lr_spec),
  "Ridge Regression"  = workflow(base_rec, ridge_spec),
  "Polynomial Ridge"  = workflow(poly_rec, ridge_spec),
  "Random Forest"     = workflow(base_rec, rf_spec),
  "XGBoost"           = workflow(base_rec, xgb_spec),
  "Neural Network"    = workflow(base_rec, nn_spec)
)

# ---- 6. Fit resamples ------------------------------------------------------
metric_set_reg <- metric_set(rsq, mae, rmse)

baseline_results <- map_dfr(names(workflows_list), function(nm) {
  cat(">> Fitting:", nm, "\n")
  res <- fit_resamples(
    workflows_list[[nm]],
    resamples = folds,
    metrics   = metric_set_reg,
    control   = control_resamples(save_pred = FALSE, verbose = FALSE)
  )
  collect_metrics(res) %>% mutate(Model = nm, .before = 1)
})

# ---- 7. Ranking table (Python format: Rank | Model | R^2 | MAE | RMSE) -----
ranking <- baseline_results %>%
  select(Model, .metric, mean) %>%
  pivot_wider(names_from = .metric, values_from = mean) %>%
  arrange(desc(rsq)) %>%
  mutate(Rank = row_number(), .before = 1) %>%
  rename(R2 = rsq, MAE = mae, RMSE = rmse)

print(ranking)

# ---- 8. Save outputs -------------------------------------------------------
readr::write_csv(ranking, file.path(out_dir, "baseline_ranking.csv"))
saveRDS(baseline_results, file.path(out_dir, "baseline_results.rds"))

cat("\nBaseline training complete. Results saved to", out_dir, "\n")
