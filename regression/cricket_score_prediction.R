# =============================================================================
# Cricket Score Prediction — Regression
# Group 8 | Reproducible Research | University of Warsaw 2026
# =============================================================================
# Authors : Rajat Dogra, Umair Aziz
# Dataset : IPL Dataset 2008-2025 (Kaggle)
# Goal    : Predict IPL cricket match scores using R (replication of ML2 Python project)
# =============================================================================

library(here)
library(tidyverse)
library(tidymodels)

# =============================================================================
# Section 2: Data Loading and EDA
# Replicates Section 2 of Cricket_Score_Prediction_Final_Submission.ipynb
# =============================================================================

# ── 2.1 Load data ─────────────────────────────────────────────────────────────
data_path <- here("regression", "data", "regression_dataset.csv")
df <- read_csv(data_path, show_col_types = FALSE)

# ── 2.2 Dimensions ────────────────────────────────────────────────────────────
cat("Dataset dimensions:", nrow(df), "rows ×", ncol(df), "features\n")
cat("Score range:", min(df$final_score), "-", max(df$final_score), "runs\n")
cat(sprintf("Average: %.1f ± %.1f\n", mean(df$final_score), sd(df$final_score)))

# ── 2.3 Data quality ──────────────────────────────────────────────────────────
cat("\nMissing values per column:\n")
print(colSums(is.na(df)))
cat("Total missing:", sum(is.na(df)), "\n")

cat("\nDuplicate rows:", sum(duplicated(df)), "\n")

cat("\nColumn types:\n")
print(sapply(df, class))

# ── 2.4 Summary statistics for final_score ────────────────────────────────────
cat("\nSummary statistics — final_score:\n")
print(summary(df$final_score))

# ── 2.5 Correlation with final_score (numeric features only) ──────────────────
numeric_cols <- df |>
  select(where(is.numeric)) |>
  select(-final_score) |>
  names()

correlations <- df |>
  select(all_of(c(numeric_cols, "final_score"))) |>
  summarise(across(
    all_of(numeric_cols),
    ~ if (sd(.x, na.rm = TRUE) == 0) NA_real_
      else cor(.x, final_score, use = "complete.obs")
  )) |>
  pivot_longer(everything(), names_to = "feature", values_to = "correlation") |>
  filter(!is.na(correlation)) |>
  mutate(abs_corr = abs(correlation)) |>
  arrange(desc(abs_corr))

cat("\nTop 5 features correlated with final_score:\n")
print(head(correlations, 5))

# ── 2.6 EDA plots ─────────────────────────────────────────────────────────────
output_dir <- here("regression", "output")

# Plot 1: Score distribution histogram
p1 <- ggplot(df, aes(x = final_score)) +
  geom_histogram(binwidth = 10, fill = "#4ECDC4", colour = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(df$final_score), linetype = "dashed", colour = "red", linewidth = 0.8) +
  labs(
    title = "Distribution of IPL Final Scores",
    subtitle = sprintf("Mean = %.1f runs  |  SD = %.1f", mean(df$final_score), sd(df$final_score)),
    x = "Final Score (runs)", y = "Count"
  ) +
  theme_minimal()

ggsave(file.path(output_dir, "01_score_distribution.png"), p1, width = 8, height = 5, dpi = 150)

# Plot 2: Team vs average score
team_avg <- df |>
  group_by(team) |>
  summarise(avg_score = mean(final_score), .groups = "drop") |>
  arrange(desc(avg_score))

p2 <- ggplot(team_avg, aes(x = reorder(team, avg_score), y = avg_score)) +
  geom_col(fill = "#FF6B6B", alpha = 0.85, colour = "white") +
  coord_flip() +
  labs(
    title = "Average Score by Team",
    x = NULL, y = "Average Final Score (runs)"
  ) +
  theme_minimal()

ggsave(file.path(output_dir, "02_team_avg_score.png"), p2, width = 9, height = 6, dpi = 150)

# Plot 3: Venue effects (top 15 venues by match count)
top_venues <- df |>
  count(venue, sort = TRUE) |>
  slice_head(n = 15) |>
  pull(venue)

venue_avg <- df |>
  filter(venue %in% top_venues) |>
  group_by(venue) |>
  summarise(avg_score = mean(final_score), n = n(), .groups = "drop")

p3 <- ggplot(venue_avg, aes(x = reorder(venue, avg_score), y = avg_score)) +
  geom_col(fill = "#45B7D1", alpha = 0.85, colour = "white") +
  geom_text(aes(label = n), hjust = -0.2, size = 3, colour = "grey40") +
  coord_flip() +
  labs(
    title = "Average Score by Venue (top 15 by match count)",
    subtitle = "Numbers = innings played at venue",
    x = NULL, y = "Average Final Score (runs)"
  ) +
  theme_minimal()

ggsave(file.path(output_dir, "03_venue_effects.png"), p3, width = 10, height = 6, dpi = 150)

# Plot 4: Wickets vs final score scatter
p4 <- ggplot(df, aes(x = wickets_lost, y = final_score)) +
  geom_jitter(alpha = 0.35, colour = "#96CEB4", width = 0.25, size = 1.2) +
  geom_smooth(method = "lm", colour = "red", se = TRUE, linewidth = 1) +
  labs(
    title = "Wickets Lost vs Final Score",
    x = "Wickets Lost", y = "Final Score (runs)"
  ) +
  theme_minimal()

ggsave(file.path(output_dir, "04_wickets_vs_score.png"), p4, width = 8, height = 5, dpi = 150)

cat("\nAll plots saved to", output_dir, "\n")
cat("Plots: 01_score_distribution.png, 02_team_avg_score.png,",
    "03_venue_effects.png, 04_wickets_vs_score.png\n")

# =============================================================================
# Section 3: Feature Engineering & Train/Test Split
# Replicates Section 3 of Cricket_Score_Prediction_Final_Submission.ipynb
# =============================================================================

# ── 3.1 Feature selection (matching Python's 12-feature set) ──────────────────
features <- c(
  "innings", "team", "wickets_lost", "balls_faced",
  "venue", "city", "toss_winner", "toss_decision",
  "year", "month", "day", "target_score"
)

df_model <- df |> select(all_of(c(features, "final_score")))

cat("\nFeature set:", length(features), "features,", nrow(df_model), "samples\n")

# ── 3.2 Train/test split — 80/20, seed 42 (matches Python) ───────────────────
set.seed(42)
split  <- initial_split(df_model, prop = 0.8)
df_train <- training(split)
df_test  <- testing(split)

cat(sprintf(
  "Training: %d rows  |  Testing: %d rows\n",
  nrow(df_train), nrow(df_test)
))
cat(sprintf(
  "Train mean: %.1f  |  Test mean: %.1f\n",
  mean(df_train$final_score), mean(df_test$final_score)
))

# ── 3.3 Preprocessing recipe ──────────────────────────────────────────────────
# Mirrors Python pipeline:
#   - target_score NAs → 0 (step_impute_median)
#   - high-cardinality categoricals collapsed (step_other)
#   - one-hot encoding (step_dummy)
#   - normalise numerics (step_normalize) — StandardScaler equivalent
#   - polynomial degree-2 on numeric predictors (step_poly)
rec <- recipe(final_score ~ ., data = df_train) |>
  step_impute_median(target_score) |>
  step_other(team, venue, city, toss_winner, threshold = 0.02) |>
  step_dummy(team, venue, city, toss_winner, toss_decision,
             one_hot = TRUE) |>
  step_normalize(all_numeric_predictors()) |>
  step_poly(innings, wickets_lost, balls_faced, year,
            degree = 2, keep_original_cols = FALSE)

rec_prep <- prep(rec, training = df_train)

train_baked <- bake(rec_prep, new_data = df_train)
test_baked  <- bake(rec_prep, new_data = df_test)

cat("\nEngineered feature count:", ncol(train_baked) - 1, "\n")
cat("Train set after baking:", nrow(train_baked), "rows\n")
cat("Test  set after baking:", nrow(test_baked),  "rows\n")

# =============================================================================
# TODO: Model training — Linear, Ridge, Lasso, Random Forest, XGBoost (issue #3)
# =============================================================================

# =============================================================================
# TODO: Hyperparameter tuning, ensembles & final evaluation (issue #4)
# =============================================================================
