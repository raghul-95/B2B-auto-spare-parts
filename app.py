import streamlit as st
import pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
import urllib.error, base64, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             r2_score, mean_absolute_error, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.signal import savgol_filter
import plotly.io as pio
pio.templates.default = "plotly_dark"
warnings.filterwarnings("ignore")

# ───────────────────────────────────────── CONFIG
DEFAULT_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto-Parts Analytics Dashboard", layout="wide")

# ──────────────────────────────────────── Data loader
@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except (urllib.error.URLError, urllib.error.HTTPError, FileNotFoundError) as e:
        st.error(f"❌ Unable to load CSV: {e}")
        st.stop()

url = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
upload = st.sidebar.file_uploader("…or upload CSV", type="csv")
df = load_csv(upload) if upload else load_csv(url)
st.sidebar.success(f"{len(df):,} rows • {df.shape[1]} cols")

# ──────────────────────────────────────── Feature engineering
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1, "No":0, "Undecided":0})

# ──────────────────────────────────────── Helpers
def make_prep(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for col in ["Lead_ID", "Company_Name"]:
        if col in cat:
            cat = cat.drop(col)
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def download_link(df_obj, filename):
    csv = df_obj.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def stringify(fset):
    return ", ".join(sorted(list(fset)))

# ──────────────────────────────────────── Tabs
tabs = st.tabs(["📊 Visualisation", "🎯 Classification",
                "🔀 Clustering", "🔗 Assoc Rules", "📈 Regression"])

# ═══════════════════════════════════════ 1  VISUALISATION
with tabs[0]:
    st.header("Interactive Descriptive Insights")
    # 1 Sales funnel
    if "Sales_Stage" in df.columns:
        counts = df["Sales_Stage"].value_counts()
        st.plotly_chart(
            go.Figure(go.Funnel(y=counts.index, x=counts.values,
                                textinfo="value+percent initial")),
            use_container_width=True
        )
    # 2 Years vs Spend with smoothed trend
    if {"Years_In_Operation","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        scatter = df[["Years_In_Operation","Monthly_Spend_Parts_INR"]].dropna().sort_values("Years_In_Operation")
        fig = px.scatter(
            df, x="Years_In_Operation", y="Monthly_Spend_Parts_INR",
            color="Business_Type" if "Business_Type" in df.columns else None,
            title="Maturity vs Monthly Spend"
        )
        if len(scatter) >= 11:
            smooth = savgol_filter(scatter["Monthly_Spend_Parts_INR"], 11, 2)
            fig.add_scatter(x=scatter["Years_In_Operation"], y=smooth,
                            mode="lines", line=dict(color="white"), name="Trend")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════ 2  CLASSIFICATION
with tabs[1]:
    st.header("Binary Classification")
    target = st.selectbox("Choose target", [c for c in ["Will_Trial_Bin","Is_Converted"] if c in df_proc.columns])
    if target:
        X, y = df_proc.drop(columns=[target]), df_proc[target]
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=42)
        prep = make_prep(X)
        learners = {"KNN": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "GBRT": GradientBoostingClassifier()}
        rows, probas = [], {}
        for name, mdl in learners.items():
            pipe = Pipeline([("prep", prep), ("est", mdl)]).fit(X_tr, y_tr)
            y_pred = pipe.predict(X_ts)
            if hasattr(pipe, "predict_proba"):
                probas[name] = pipe.predict_proba(X_ts)[:,1]
            rep = classification_report(y_ts, y_pred, output_dict=True, zero_division=0)
            rows.append({"Model": name,
                         "Accuracy": round(rep["accuracy"], 3),
                         "Precision": round(rep["weighted avg"]["precision"], 3),
                         "Recall": round(rep["weighted avg"]["recall"], 3),
                         "F1": round(rep["weighted avg"]["f1-score"], 3)})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

        # Confusion matrix toggle
        cm_mod = st.selectbox("Confusion-matrix model", list(learners.keys()))
        if st.button("Show CM"):
            cm_pipe = Pipeline([("prep", prep), ("est", learners[cm_mod])]).fit(X_tr, y_tr)
            cm = confusion_matrix(y_ts, cm_pipe.predict(X_ts))
            st.plotly_chart(
                px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"),
                          title=f"Confusion Matrix – {cm_mod}"),
                use_container_width=True
            )

        # ROC curves
        if st.checkbox("Show ROC curves"):
            fig = go.Figure()
            for n, proba in probas.items():
                fpr, tpr, _ = roc_curve(y_ts, proba)
                fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                         mode="lines",
                                         name=f"{n} (AUC {auc(fpr,tpr):.2f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                     mode="lines", line=dict(dash="dash"),
                                     showlegend=False))
            fig.update_layout(title="ROC Curves",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════ 3  CLUSTERING
with tabs[2]:
    st.header("K-Means Clustering")
    k = st.slider("k", 2, 10, 4)
    num_df = df_proc.select_dtypes(include="number").fillna(df_proc.select_dtypes(include="number").median())
    scaled = StandardScaler().fit_transform(num_df)
    inertias = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(scaled).inertia_
                for i in range(2, 11)]
    st.plotly_chart(px.line(x=list(range(2,11)), y=inertias,
                            markers=True, title="Elbow Chart"),
                    use_container_width=True)

    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(scaled)
    df_clust = df_proc.copy()
    df_clust["cluster"] = km.labels_
    st.dataframe(df_clust.groupby("cluster")[["Monthly_Spend_Parts_INR"]].mean().round(1))
    st.markdown(download_link(df_clust, "clustered_data.csv"), unsafe_allow_html=True)

# ═══════════════════════════════════════ 4  ASSOC RULES
with tabs[3]:
    st.header("Association Rule Mining")
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols) >= 2:
        colA = st.selectbox("Column A", cat_cols)
        colB = st.selectbox("Column B", [c for c in cat_cols if c != colA])
        min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)
        if st.button("Run Apriori"):
            basket = pd.get_dummies(df[[colA, colB]])
            freq = apriori(basket, min_support=min_sup, use_colnames=True)
            if freq.empty:
                st.warning("No itemsets meet support threshold.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                if rules.empty:
                    st.warning("No rules meet confidence threshold.")
                else:
                    rules["antecedents"] = rules["antecedents"].apply(stringify)
                    rules["consequents"] = rules["consequents"].apply(stringify)
                    st.dataframe(rules.sort_values("lift", ascending=False)
                                 .head(10)[["antecedents","consequents",
                                            "support","confidence","lift"]],
                                 use_container_width=True)
    else:
        st.info("Need at least two categorical columns.")

# ═══════════════════════════════════════ 5  REGRESSION
with tabs[4]:
    st.header("Regression")
    target_num = st.selectbox("Numeric target",
                              [c for c in ["Expected_Monthly_Spend_NewVendor_INR",
                                           "Monthly_Spend_Parts_INR"] if c in df_proc.columns])
    if target_num:
        Xn, yn = df_proc.drop(columns=[target_num]), df_proc[target_num]
        Xtr, Xts, ytr, yts = train_test_split(Xn, yn, test_size=0.2, random_state=42)
        prep_num = make_prep(Xn)
        regs = {"Linear": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.1),
                "Decision Tree": DecisionTreeRegressor(random_state=42)}
        rows = []
        for name, reg in regs.items():
            pipe = Pipeline([("prep", prep_num), ("reg", reg)]).fit(Xtr, ytr)
            preds = pipe.predict(Xts)
            rows.append({"Model": name,
                         "R²": round(r2_score(yts, preds), 3),
                         "MAE": round(mean_absolute_error(yts, preds), 1),
                         "RMSE": round(np.sqrt(mean_squared_error(yts, preds)), 1)})
        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metrics_df)

        sel = st.selectbox("Show Actual vs Predicted for", metrics_df.index)
        best_pipe = Pipeline([("prep", prep_num), ("reg", regs[sel])]).fit(Xtr, ytr)
        preds = best_pipe.predict(Xts)
        fig = px.scatter(x=yts, y=preds,
                         labels=dict(x="Actual", y="Predicted"),
                         title=f"Actual vs Predicted – {sel}")
        min_, max_ = yts.min(), yts.max()
        fig.add_trace(go.Scatter(x=[min_, max_], y=[min_, max_],
                                 mode="lines", line=dict(dash="dash"),
                                 showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
""")

with open(os.path.join(bundle_dir,"app.py"), "w") as f:
    f.write(app_py)

# Re-zip
zip_path = "/mnt/data/auto_parts_dashboard.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(bundle_dir):
        for file in files:
            z.write(os.path.join(root, file),
                    arcname=os.path.relpath(os.path.join(root, file), bundle_dir))
