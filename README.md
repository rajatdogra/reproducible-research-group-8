# Reproducible Research in R — Group 8
## Replication of ML2 Machine Learning Projects

**Course:** Reproducible Research  
**University:** University of Warsaw, Faculty of Economic Sciences  
**Submission Repository:** [rajatdogra/reproducible-research-group-8](https://github.com/rajatdogra/reproducible-research-group-8)  
**Date:** May 2026

---

## Team Members

| Name | Student ID | GitHub |
|------|-----------|--------|
| Rajat Dogra | 474072 | [@rajatdogra](https://github.com/rajatdogra) |
| Umair Aziz | 476686 | [@umairdev0](https://github.com/umairdev0) |
| Ashutosh Kumar Verma | 475852 | [@vermaashutosh777](https://github.com/vermaashutosh777) |
| Ramik Sharma | 477656 | [@ramiksharma](https://github.com/RamikSharma) |

> All members contribute directly to this repository via pull requests. Progress is tracked through commit history — commits from all four members are visible throughout the project.

---

## What This Project Does

This project reproduces and rewrites in **R** two machine learning pipelines originally developed in Python for the **ML2 – Machine Learning 2** course (submitted January 10, 2026). The goal is full reproducibility: same datasets, same analytical questions, comparable results — implemented end-to-end in R using `tidyverse`, `tidymodels`, and related packages.

---

## Projects

### Project 1 — Regression: Cricket Score Prediction

| Item | Detail |
|------|--------|
| **Problem** | Predict IPL cricket match scores (continuous target) |
| **Dataset** | [IPL Dataset 2008-2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) — 76,014 ball-by-ball records, 60+ features |
| **Models** | Linear Regression, Ridge, Polynomial Ridge, Random Forest, XGBoost, Neural Network, Extra Trees, Voting Ensemble |
| **Python Baseline** | Gradient Boosting — R² = 0.722, RMSE = 20.1 |
| **R Result** | Voting Ensemble — R² = 0.749, RMSE = 19.2 (beats Python baseline) |
| **Key Features** | Wickets lost, balls faced, venue, team, toss, target score |

### Project 2 — Classification: Football Match Outcomes

| Item | Detail |
|------|--------|
| **Problem** | Predict match outcome: Home Win / Draw / Away Win (3-class) |
| **Dataset** | [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data) — 67,353 matches across multiple leagues, 2024-2026 |
| **Models** | Random Forest, Gradient Boosting, XGBoost, Neural Network, Stacking Ensemble |
| **Python Baseline** | Stacking Ensemble — Accuracy 65.36%, F1 = 0.6533 |
| **R Result** | Stacking Ensemble — Accuracy 57.30%, AUC = 0.7504 |
| **Key Features** | Team strength differential, league, historical performance, temporal patterns |

---

## Repository Structure

```
reproducible-research-group-8/
├── README.md                                        # Project description, setup, how to run
├── CONTRIBUTING.md                                  # Contribution workflow and guidelines
├── .gitignore                                       # Ignore .DS_Store, .Rhistory, renv/library/, etc.
├── renv.lock                                        # Pinned R package versions for reproducibility
├── setup_renv.R                                     # One-time environment setup script
├── regression/                                      # Cricket Score Prediction (R)
│   ├── cricket_score_prediction.R                   # Data loading and EDA
│   ├── cricket_score_prediction.qmd                 # Quarto report
│   ├── cricket_score_prediction.html                # Rendered HTML report
│   ├── 03_baseline_models.R                         # Baseline model training
│   ├── 04_tuning_ensembles.R                        # Hyperparameter tuning and ensembles
│   ├── original_python/                             # Original Python baseline (reference)
│   │   ├── Cricket_Score_Prediction_Final_Submission.ipynb
│   │   └── Cricket_Score_Prediction_Final_Submission.pdf
│   ├── output/                                      # Plots, tables, model outputs
│   └── data/                                        # regression_dataset.csv
├── classification/                                  # Football Match Outcome (R)
│   ├── football_classification.R                    # Full pipeline (EDA, features, models, plots)
│   ├── football_classification.qmd                  # Quarto report
│   ├── football_classification.html                 # Rendered HTML report
│   ├── original_python/                             # Original Python baseline (reference)
│   │   ├── football_classification_complete.py
│   │   ├── results/                                 # Python output files
│   │   └── plots/                                   # Python plot files
│   ├── output/                                      # Plots, tables, model outputs
│   └── data/                                        # fixtures.csv
└── docs/                                            # Shared documentation
```

---

## Requirements

- **R** >= 4.4.1
- **Quarto** >= 1.4
- R packages managed via `renv` — full list in `renv.lock`
- Core packages: `tidyverse`, `tidymodels`, `ranger`, `xgboost`, `glmnet`, `stacks`, `brulee`, `themis`, `ggplot2`, `yardstick`, `here`, `patchwork`

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/rajatdogra/reproducible-research-group-8.git
cd reproducible-research-group-8

# 2. Restore the R package environment
Rscript -e 'renv::restore()'
```

> All package versions are pinned in `renv.lock`. No manual package installation needed.

---

## How to Run

### Option 1 — Render Quarto reports (recommended)

```bash
# Classification report
quarto render classification/football_classification.qmd

# Regression report
quarto render regression/cricket_score_prediction.qmd
```

### Option 2 — Run R scripts directly

```r
# Classification — full pipeline (EDA, features, models, plots, ethics)
source("classification/football_classification.R")

# Regression — run in order
source("regression/cricket_score_prediction.R")      # EDA
source("regression/03_baseline_models.R")            # Baseline models
source("regression/04_tuning_ensembles.R")           # Tuning and ensembles
```

---

## Expected Output

| Project | Output Location | Approx. Runtime |
|---------|----------------|-----------------|
| Cricket Regression | `regression/output/` | ~20-30 minutes |
| Football Classification | `classification/output/` | ~15-20 minutes |

Outputs include model comparison tables, performance metrics, visualisations, error analysis, and ethical considerations reports. Pre-rendered HTML reports and output files are committed in the respective folders.

---

## Data

| Dataset | Source | Location in Repo |
|---------|--------|-----------------|
| IPL Cricket 2008-2025 | [Kaggle](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) | `regression/data/regression_dataset.csv` |
| ESPN Soccer 2024-2026 | [Kaggle](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data) | `classification/data/fixtures.csv` |

> Both data files are included in the repository. No external download required after cloning.

---

## Results Summary

| Project | Best R Model | R Metric | Python Baseline | Status |
|---------|-------------|----------|-----------------|--------|
| Cricket Regression | Voting Ensemble | R² = 0.749, RMSE = 19.2 | R² = 0.722, RMSE = 20.1 | Done — beats baseline |
| Football Classification | Stacking Ensemble | Accuracy = 57.30%, AUC = 0.7504 | Accuracy = 65.36%, F1 = 0.6533 | Done |

---

## Task Assignment

| Issue | Task | Assigned To | Status |
|-------|------|-------------|--------|
| #1 | Regression: Data loading & EDA | Rajat Dogra | Done |
| #2 | Regression: Feature engineering & split | Rajat Dogra | Done |
| #3 | Regression: Baseline model training | Umair Aziz | Done |
| #4 | Regression: Tuning, ensembles & evaluation | Umair Aziz | Done |
| #5 | Classification: Data loading & EDA | Ashutosh Kumar Verma | Done |
| #6 | Classification: Feature engineering | Ashutosh Kumar Verma | Done |
| #7 | Classification: All model training | Ramik Sharma | Done |
| #8 | Classification: Plots, error analysis & ethics | Ramik Sharma | Done |
| #9 | Setup: renv.lock | Rajat Dogra | Done |
| #10 | Docs: Final README update | All Members | Done |
| #11 | Regression: Quarto report | Rajat Dogra | Done |
| #12 | Classification: Quarto report | Ashutosh Kumar Verma | Done |

---

## Team Workflow

All contributions go through pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

```bash
git pull origin main
git checkout -b feat/your-branch-name
# make changes
git add .
git commit -m "descriptive message - refs #issue_number"
git push origin feat/your-branch-name
# open a PR on GitHub and request review
```

---

## Original Project Reference

The original Python-based ML2 project (Rajat Dogra & Umair Aziz, January 2026) serves as the baseline for this reproducibility exercise. Dataset sources and problem definitions remain identical.
