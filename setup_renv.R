# Run this script once after cloning to initialise renv
# Rscript setup_renv.R

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")

renv::init()

# Core packages for this project
packages <- c(
  "tidyverse",
  "tidymodels",
  "ranger",       # Random Forest
  "xgboost",      # XGBoost
  "glmnet",       # Ridge / Lasso
  "stacks",       # Stacking Ensemble
  "ggplot2",      # Visualisation
  "yardstick",    # Model metrics
  "here",         # Relative paths
  "quarto"        # Report rendering
)

install.packages(packages)
renv::snapshot()

cat("\nSetup complete. renv.lock has been created.\n")
cat("Share renv.lock with your team so everyone uses identical package versions.\n")
