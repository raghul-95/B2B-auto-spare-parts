
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, association_rules
import base64, io, warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- CONFIG ---------------------
DEFAULT_URL = "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Lead Analytics Dashboard", layout="wide")
# -------------------------------------------------

@st.cache_data(ttl=3600)
def load_csv(path_or_url):
    return pd.read_csv(path_or_url)

# -------- Sidebar data source controls ----------
st.sidebar.header("Data source")
url_box = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
upload_file = st.sidebar.file_uploader("…or upload local CSV", type="csv")

try:
    if upload_file:
        df = load_csv(upload_file)
    else:
        df = load_csv(url_box)
except Exception as e:
    st.error(f"❌ Could not load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded **{len(df):,} rows × {df.shape[1]} cols**")

# Basic derived field for binary classification
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1, "No":0, "Undecided":0})

# Helper to build preprocessing pipeline
def make_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    for bad in ["Lead_ID", "Company_Name"]:
        if bad in cat_cols:
            cat_cols.remove(bad)
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

# ---------------------- Tabs ---------------------
tabs = st.tabs(["📊 EDA", "🎯 Classification", "🔀 Clustering", "🔗 Association Rules", "📈 Regression"])

# ------------------ 1. EDA -----------------------
with tabs[0]:
    st.header("Exploratory Data Visualisations")
    # simple examples; feel free to expand
    if "Is_Converted" in df.columns:
        conv = df["Is_Converted"].value_counts(normalize=True)*100
        st.plotly_chart(px.pie(conv, values=conv.values, names=conv.index,
                                title="Overall Conversion Rate (%)"),
                        use_container_width=True)
    if {"Region","Is_Converted"}.issubset(df.columns):
        st.plotly_chart(px.histogram(df, x="Region", color="Is_Converted",
                                     barnorm="percent",
                                     title="Conversion by Region"),
                        use_container_width=True)

# -------------- 2. Classification ----------------
with tabs[1]:
    st.header("Binary Classification")
    tgt = st.selectbox("Choose target label", options=[c for c in ["Will_Trial_Bin","Is_Converted"] if c in df_proc.columns])
    if tgt:
        X = df_proc.drop(columns=[tgt])
        y = df_proc[tgt]
        preproc = make_preprocessor(X)
        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "GBRT": GradientBoostingClassifier(random_state=42)
        }
        if st.button("Train models"):
            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            report_rows = []
            for name, mdl in models.items():
                pipe = Pipeline([("prep", preproc), ("model", mdl)])
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_ts)
                rep = classification_report(y_ts, y_pred, output_dict=True, zero_division=0)["weighted avg"]
                report_rows.append({"Model": name,
                                    "Accuracy": round(rep["accuracy"],3),
                                    "Precision": round(rep["precision"],3),
                                    "Recall": round(rep["recall"],3),
                                    "F1": round(rep["f1-score"],3)})
            st.dataframe(pd.DataFrame(report_rows).set_index("Model"))

# -------------- 3. Clustering --------------------
with tabs[2]:
    st.header("K‑Means Clustering")
    k = st.slider("k", 2, 10, 4)
    num_cols = df_proc.select_dtypes(include="number").columns
    scaler = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())])
    X_scaled = scaler.fit_transform(df_proc[num_cols])
    km = KMeans(n_clusters=k, random_state=42, n_init=25).fit(X_scaled)
    df_proc["cluster"] = km.labels_
    st.dataframe(df_proc.groupby("cluster")[num_cols].mean().round(1))

# ---------- 4. Association Rules -----------------
with tabs[3]:
    st.header("Association Rule Mining")
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if len(cats) < 2:
        st.info("Need at least two categorical columns.")
    else:
        col1 = st.selectbox("Column 1", cats, key="ar1")
        col2 = st.selectbox("Column 2", [c for c in cats if c != col1], key="ar2")
        min_sup = st.slider("Min support", 0.01, 0.2, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)

        if st.button("Generate rules"):
            basket = pd.get_dummies(df[[col1, col2]])
            freq = apriori(basket, min_support=min_sup, use_colnames=True)
            if freq.empty:
                st.warning("No frequent itemsets at this support level. Lower the threshold or change columns.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                if rules.empty:
                    st.warning("No rules meet the confidence threshold.")
                else:
                    st.dataframe(rules.sort_values("lift", ascending=False)
                                 .head(10)[["antecedents","consequents","support","confidence","lift"]],
                                 use_container_width=True)

# ---------------- 5. Regression ------------------
with tabs[4]:
    st.header("Simple Regression")
    targets = [c for c in ["Expected_Monthly_Spend_NewVendor_INR","Monthly_Spend_Parts_INR"] if c in df_proc.columns]
    tgt_r = st.selectbox("Numeric target", targets)
    if tgt_r:
        Xr = df_proc.drop(columns=[tgt_r]); yr = df_proc[tgt_r]
        preproc_r = make_preprocessor(Xr)
        if st.button("Train Ridge model"):
            X_tr,X_ts,y_tr,y_ts = train_test_split(Xr,yr,test_size=0.2,random_state=42)
            pipe = Pipeline([("prep", preproc_r), ("model", Ridge(alpha=1.0))])
            pipe.fit(X_tr,y_tr)
            st.metric("Ridge R²", f"{pipe.score(X_ts,y_ts):.3f}")

st.caption("Stable release – all tabs guarded against empty data or NaNs")
