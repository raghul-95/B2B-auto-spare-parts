# B2B Auto‑Parts Lead‑Analytics Dashboard – Pro Edition

This version meets the **full professor rubric**:

* **Data Visualisation**: 10+ interactive, business‑focused charts with narrative captions.
* **Classification**: KNN, Decision Tree, Random Forest, GBRT – metric table, confusion‑matrix toggle, unified ROC curve, batch prediction upload & download.
* **Clustering**: k‑slider, elbow chart, persona summary table, downloadable labelled data.
* **Association Rules**: Apriori with user‑adjustable columns/support/confidence; top‑10 lift‑sorted rules.
* **Regression**: Linear, Ridge, Lasso, DecisionTreeRegressor with seven KPIs (R², MAE, etc.) and feature‑importance bar.

All charts are built with Plotly for C‑suite polish; descriptive captions explain the takeaway.

## Deploy steps
1. Edit `DEFAULT_URL` in **app.py** with your raw CSV link.  
2. Push `app.py`, `requirements.txt`, `README.md` to GitHub.  
3. In Streamlit Cloud, set main file to `app.py` and deploy.

