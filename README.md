
# B2B Auto‑Parts Lead‑Analytics Dashboard

This Streamlit app delivers interactive EDA, classification, clustering,
association‑rule mining, and regression for the synthetic spare‑parts
lead‑pipeline dataset.

## Auto‑loading data

`app.py` is pre‑configured with a **hard‑wired GitHub raw URL** so the dashboard
opens fully populated—no sidebar copy‑past needed.

Edit the `DEFAULT_URL` constant near the top of *app.py*:

```python
DEFAULT_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/Data_Analysis_R_Survey_Enhanced.csv"
```

Commit & push—Streamlit Cloud will always fetch that file on start‑up.  
The sidebar widgets remain, so you can still upload or paste a different CSV
for ad‑hoc “what‑if” analyses.

## Repo structure

```
├── app.py            # Main Streamlit entry
├── requirements.txt  # Dependencies
└── README.md
```

## Deploy on Streamlit Cloud

1. Push these three files plus your CSV to a GitHub repo.
2. In **Streamlit Cloud ➜ New App**, point to `app.py` on `main`.
3. (Optional) delete any `GITHUB_RAW_URL` secret—`DEFAULT_URL` is baked in.
