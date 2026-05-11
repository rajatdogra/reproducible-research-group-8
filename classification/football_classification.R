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
  "quarter", "is_holiday_season",
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

# Add outcome (target)
df_model <- df %>%
  select(all_of(model_features), outcome) %>%
  mutate(outcome = factor(outcome,
                          levels = c("Home Win", "Draw", "Away Win")))

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
  step_dummy(all_nominal_predictors()) %>%
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