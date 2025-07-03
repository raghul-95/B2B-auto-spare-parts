
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import base64, warnings
warnings.filterwarnings('ignore')
import urllib.error  # <- at top with the imports

@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        st.error(f"❌ Unable to fetch CSV from URL – {e}. "
                 "Check that you used the **Raw** link or upload the file instead.")
        st.stop()

DEFAULT_URL = "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Analytics Pro‑v6", layout="wide")

@st.cache_data(ttl=3600)
def load_csv(src): return pd.read_csv(src)

url = st.sidebar.text_input("GitHub raw CSV URL", DEFAULT_URL)
up = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_csv(up) if up else load_csv(url)
st.sidebar.success(f"{len(df):,} rows loaded")

df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1,"No":0,"Undecided":0})

def preproc(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for drop in ["Lead_ID","Company_Name"]:
        if drop in cat: cat = cat.drop(drop)
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("sc",StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def dl_link(dataframe, fname):
    csv = dataframe.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{fname}">Download CSV</a>'

tabs = st.tabs(["EDA","Classification","Clustering","Assoc Rules","Regression"])

# ----- EDA -----
with tabs[0]:
    st.header("Interactive Descriptive Insights")
    # Robust Sales Stage Funnel
    if "Sales_Stage" in df.columns:
        stage_counts = df["Sales_Stage"].value_counts()
        fig = go.Figure(go.Funnel(
            y=stage_counts.index.tolist(),
            x=stage_counts.values.tolist(),
            textposition="inside",
            textinfo="value+percent previous"
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Sales funnel using actual Sales_Stage categories.")

    # Years vs Spend scatter without statsmodels dependency
    if {"Years_In_Operation","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        fig = px.scatter(df, x="Years_In_Operation", y="Monthly_Spend_Parts_INR",
                         color="Business_Type" if "Business_Type" in df.columns else None,
                         trendline="lowess", title="Maturity vs Spend (LOWESS)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Relationship between years active and spend (LOWESS trend).")

# ---- Classification minimal demonstration (same logic) --
with tabs[1]:
    st.header("Classification demo")
    if "Is_Converted" in df_proc.columns:
        X,y = df_proc.drop(columns=["Is_Converted"]), df_proc["Is_Converted"]
        Xtr,Xts,ytr,yts = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        models = {"RF":RandomForestClassifier()}
        rows=[]
        pp = preproc(X)
        for n,m in models.items():
            pipe=Pipeline([("p",pp),("m",m)]).fit(Xtr,ytr)
            ypred=pipe.predict(Xts)
            rep=classification_report(yts, ypred, output_dict=True)
            rows.append({"Model":n,
                         "Accuracy":round(rep["accuracy"],3),
                         "F1":round(rep["weighted avg"]["f1-score"],3)})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))
