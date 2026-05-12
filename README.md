# Reproducible Research in R — Group 8
## Replication of ML2 Machine Learning Projects

**Course:** Reproducible Research  
**University:** University of Warsaw, Faculty of Economic Sciences  
**Submission Repository:** [rajatdogra/reproducible-research-group-8](https://github.com/rajatdogra/reproducible-research-group-8)  
**Date:** May/13/2026

---

## Team Members

| Name | Student ID | GitHub |
|------|-----------|--------|
| Rajat Dogra | 474072 | [@rajatdogra](https://github.com/rajatdogra) |
| Umair Aziz | 476686 | [@umairdev0](https://github.com/umairdev0) |
| Ashutosh Kumar Verma | 475852 | [@vermaashutosh777](https://github.com/vermaashutosh777) |
| Ramik Sharma | 477656 | [@ramiksharma](https://github.com/RamikSharma)

> All members contribute directly to this repository. Progress is tracked through commit history — commits from all four members are expected throughout the project.

---

## What This Project Does

This project reproduces and rewrites in **R** two machine learning pipelines originally developed in Python for the **ML2 – Machine Learning 2** course (submitted January 10, 2026). The goal is full reproducibility: same datasets, same analytical questions, comparable results — implemented end-to-end in R using `tidyverse`, `tidymodels`, and related packages.

---

## Projects

### Project 1 — Regression: Cricket Score Prediction

| Item | Detail |
|------|--------|
| **Problem** | Predict IPL cricket match scores (continuous target) |
| **Dataset** | [IPL Dataset 2008–2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) — 76,014 ball-by-ball records, 60+ features |
| **Models** | Linear Regression, Ridge, Lasso, Random Forest, XGBoost, and others |
| **Python Baseline** | Random Forest — R² = 0.89, RMSE = 12.4 runs |
| **Key Features** | Current run rate, wickets fallen, venue, team performance |

### Project 2 — Classification: Football Match Outcomes

| Item | Detail |
|------|--------|
| **Problem** | Predict match outcome: Home Win / Draw / Away Win (3-class) |
| **Dataset** | [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data) — 67,353 matches across multiple leagues, 2024–2026 |
| **Models** | Random Forest, XGBoost, Neural Networks, Stacking Ensemble |
| **Python Baseline** | Stacking Ensemble — Accuracy 65.36%, F1 = 0.6533 |
| **Key Features** | Team strength differential, league, historical performance, temporal patterns |

---

##  Repository Structure

```
reproducible-research-group-8/
├── README.md                             # Project description, setup, how to run
├── .gitignore                            # Ignore .DS_Store, .Rhistory, renv/library/, etc.
├── renv.lock                             # Pinned R package versions for reproducibility
├── regression/                           # Cricket Score Prediction (R)
│   ├── cricket_score_prediction.R        # Main R script
│   ├── cricket_score_prediction.Rmd      # R Markdown / Quarto report
│   ├── output/                           # Rendered plots, tables, reports
│   └── data/                             # regression_dataset.csv
├── classification/                       # Football Match Outcome (R)
│   ├── football_classification.R         # Main R script
│   ├── football_classification.Rmd       # R Markdown / Quarto report
│   ├── output/                           # Rendered plots, tables, reports
│   └── data/                             # fixtures.csv
└── docs/                                 # Shared documentation
```

---

## Requirements

- **R** >= 4.3
- **Quarto** >= 1.4 (for rendering reports)
- R packages managed via `renv` — full list in `renv.lock`
- Core packages: `tidyverse`, `tidymodels`, `ranger`, `xgboost`, `glmnet`, `stacks`, `ggplot2`, `yardstick`, `here`

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/rajatdogra/reproducible-research-group-8.git
cd reproducible-research-group-8

# 2. Open R and restore the package environment
Rscript -e 'renv::restore()'
```

> All package versions are pinned in `renv.lock`. No manual package installation needed.

---

## How to Run

```bash
# Render the regression report
quarto render regression/cricket_score_prediction.Rmd

# Render the classification report
quarto render classification/football_classification.Rmd
```

Or run the scripts directly in R:

```r
# Issues #5 and #6: EDA and feature engineering
# Issues #7 and #8: model training, evaluation, plots, error analysis
source("classification/football_classification.R")
```

---

## Expected Output

| Project | Output Location | Approx. Runtime |
|---------|----------------|-----------------|
| Cricket Regression | `regression/output/` | ~5–10 minutes |
| Football Classification | `classification/output/` | ~10–15 minutes |

Outputs include model comparison tables, performance metrics, and visualisations. Pre-rendered versions are committed in the respective `output/` folders.

---

## Data

| Dataset | Source | Location in Repo |
|---------|--------|-----------------|
| IPL Cricket 2008–2025 | [Kaggle](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) | `regression/data/regression_dataset.csv` |
| ESPN Soccer 2024–2026 | [Kaggle](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data) | `classification/data/fixtures.csv` |

> Data files are included in the repository. No external download required after cloning.

---

## Team Workflow

1. All team members/contributors - MUST follow the workflow in CONTRIBUTING.md: Create a branch, submit a PR, and wait for review before merging to main in this repository.
2. Always `git pull` before starting work to avoid conflicts
3. Use meaningful commit messages (e.g. `"add feature engineering step for cricket run rate"` not `"update"`)
4. Commits from all four members are expected and will be reviewed by the instructor

```bash
git pull origin main          # pull latest before working
git checkout -b feat/7-classification-model-training
# ... make changes ...
git add .
git commit -m "feat: descriptive message about what changed"
git push origin feat/7-classification-model-training
# then open a PR on GitHub referencing Closes #7
git push origin main
```

---

## Results Summary

| Project | Best Model (Python) | Metric | R Replication Status |
|---------|-------------------|--------|----------------------|
| Cricket Regression | Random Forest | R² = 0.89, RMSE = 12.4 runs | In progress |
| Football Classification | Stacking Ensemble | Accuracy = 65.36%, F1 = 0.6533 | See classification/output/comprehensive_results.txt |

> This table will be updated as R results are produced.

---

## Original Project Reference used for Reproducible Research

The original Python-based ML2 project (Rajat Dogra & Umair Aziz, January 2026) serves as the baseline for this reproducibility exercise. Dataset sources and problem definitions remain identical.
