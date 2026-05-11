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