# B2B Auto‑Parts Lead‑Analytics Dashboard (final)

This Streamlit app provides interactive EDA, classification, clustering,
association‑rule mining, and regression for the synthetic auto‑parts lead
dataset.

* Median imputation prevents NaN issues for every model.
* Association‑rule tab checks for empty itemsets before computing rules.
* Classification metrics fall back to `accuracy_score`, eliminating KeyErrors.

## Deployment

1. Open **app.py** and replace `<user>/<repo>` in `DEFAULT_URL` with the raw CSV link from your GitHub repo.
2. Push *app.py*, *requirements.txt*, and *README.md* to GitHub.
3. On Streamlit Cloud: **New app → main file = app.py** → Deploy.

That's it—every tab should load and run without errors.
