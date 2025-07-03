
# B2B Auto‑Parts Lead‑Analytics Dashboard (stable release)

This version includes robust error‑handling so **no tab can crash**:

* NaNs are imputed (median) before any model training.
* Association‑rule mining checks for empty itemsets **before** calling `association_rules`.
* All optional heavy computations are gated behind buttons.
* Defaults are set but every sidebar input can be overridden.

## Deploy

1. Replace `<user>/<repo>` in `DEFAULT_URL` in *app.py* with your GitHub raw CSV.
2. Push `app.py`, `requirements.txt`, and this README to your repo.
3. Deploy on Streamlit Cloud (main file = `app.py`).

