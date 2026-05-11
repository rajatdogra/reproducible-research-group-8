# Contributing

## Branch naming

```
feat/<issue-number>-short-description
fix/<issue-number>-short-description
docs/<issue-number>-short-description
chore/<issue-number>-short-description
```

Examples:
- `feat/1-regression-eda`
- `docs/10-readme-update`

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add regression EDA visualisations
fix: correct missing value handling in preprocessing
docs: update README with final results
chore: update renv.lock
```

## Workflow

1. Create a branch off `main` using the naming convention above
2. Make your changes, committing incrementally with descriptive messages
3. Open a PR referencing the issue (`Closes #N`)
4. Request a review before merging
5. Merge only after approval — never push directly to `main`
