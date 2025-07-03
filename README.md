# B2B Auto‑Parts Lead‑Analytics Dashboard

This Streamlit app lets you explore, model and segment the synthetic lead‑pipeline data
for a new automotive spare‑parts distributor.

## Repository layout

```
├── app.py                 # Main Streamlit entry point
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick start on Streamlit Cloud

1. **Create a new repo** on GitHub (public or private).
2. Drop these three files into the repo & push.
3. In [Streamlit Cloud](https://share.streamlit.io/):
   * **New app → Connect to GitHub** → select the repo.
   * Main file: `app.py` & branch: `main`.
4. In *Advanced settings* → **Secrets** add

   ```toml
   GITHUB_RAW_URL = "https://raw.githubusercontent.com/<user>/<repo>/<branch>/Data_Analysis_R_Survey_Enhanced.csv"
   ```

5. Click **Deploy** – your multi‑tab dashboard will spin up.

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data source

The app pulls the enriched dataset (**`Data_Analysis_R_Survey_Enhanced.csv`**)
from the GitHub raw URL.  
You can also upload a local CSV via the sidebar.

---
