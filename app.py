
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
import base64, io

DEFAULT_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/Data_Analysis_R_Survey_Enhanced.csv"

st.set_page_config(page_title="Auto‑Parts Lead Analytics", layout="wide")

@st.cache_data(ttl=3600)
def load_csv(src):
    return pd.read_csv(src)

st.sidebar.header("Data source")
url_input = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
uploaded_file = st.sidebar.file_uploader("…or upload local CSV", type="csv")

if uploaded_file is not None:
    df = load_csv(uploaded_file)
elif url_input:
    df = load_csv(url_input)
else:
    st.error("Provide CSV")
    st.stop()

st.sidebar.success(f"{len(df):,} rows • {df.shape[1]} cols")

binary_map = {"Yes":1,"No":0,"Undecided":0}
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map(binary_map)

num_cols_all = df_proc.select_dtypes(include="number").columns.tolist()
cat_cols_all = df_proc.select_dtypes(exclude="number").columns.tolist()
for c in ["Lead_ID","Company_Name"]:
    if c in cat_cols_all:
        cat_cols_all.remove(c)

tab_titles = ["📊 Visualisation","🎯 Classification","🔀 Clustering","🔗 Assoc Rules","📈 Regression"]
tabs = st.tabs(tab_titles)

# 1 Visualisation (unchanged minimal)
with tabs[0]:
    st.header("Quick EDA")
    if "Is_Converted" in df.columns:
        conv = df["Is_Converted"].value_counts(normalize=True)*100
        st.plotly_chart(px.pie(conv, values=conv.values, names=conv.index,
                                title="Lead Conversion Split"), use_container_width=True)

# ---- Helper: build preprocessing pipeline with imputer ----
def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

# 2 Classification
with tabs[1]:
    st.header("Classification")
    target = st.selectbox("Target", ["Will_Trial_Bin","Is_Converted"])
    if target not in df_proc.columns:
        st.warning("Target not present.")
    else:
        X = df_proc.drop(columns=[target])
        y = df_proc[target]
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        preproc = build_preprocessor(num_cols, cat_cols)
        models = {"KNN":KNeighborsClassifier(),
                  "DT":DecisionTreeClassifier(random_state=42),
                  "RF":RandomForestClassifier(random_state=42),
                  "GBRT":GradientBoostingClassifier(random_state=42)}
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,
                                               stratify=y,random_state=42)
        res = {}
        for n,m in models.items():
            pipe=Pipeline([("prep",preproc),("m",m)])
            pipe.fit(X_tr,y_tr)
            y_pred = pipe.predict(X_te)
            rep=classification_report(y_te,y_pred,output_dict=True,zero_division=0)
            res[n]=rep["weighted avg"]; res[n]["accuracy"]=rep["accuracy"]
        st.dataframe(pd.DataFrame(res).T.round(3))

# 3 Clustering
with tabs[2]:
    st.header("K‑Means Clustering")
    k = st.slider("k",2,10,4)
    num_cols = df_proc.select_dtypes(include="number").columns
    scaler = Pipeline([("impute",SimpleImputer(strategy="median")),
                       ("scale",StandardScaler())])
    X_scaled = scaler.fit_transform(df_proc[num_cols])
    km = KMeans(n_clusters=k,random_state=42,n_init=25).fit(X_scaled)
    df_proc["cluster"]=km.labels_
    st.dataframe(df_proc.groupby("cluster")[num_cols].mean().round(1))

# 4 Association rules (unchanged minimal)
with tabs[3]:
    st.header("Association Rules")
    cats=df_proc.select_dtypes(exclude="number").columns.tolist()
    if len(cats)>=2:
        c1=st.selectbox("A",cats,key="a")
        c2=st.selectbox("B",[c for c in cats if c!=c1],key="b")
        basket=pd.get_dummies(df[[c1,c2]])
        rules=association_rules(apriori(basket,min_support=0.05,use_colnames=True),
                                metric="confidence",min_threshold=0.3)
        if not rules.empty:
            st.dataframe(rules.head(10)[["antecedents","consequents","support","confidence","lift"]])

# 5 Regression
with tabs[4]:
    st.header("Regression")
    target_r=st.selectbox("Numeric target",["Expected_Monthly_Spend_NewVendor_INR",
                                            "Monthly_Spend_Parts_INR"])
    Xr=df_proc.drop(columns=[target_r]); yr=df_proc[target_r]
    num_cols=Xr.select_dtypes(include="number").columns.tolist()
    cat_cols=Xr.select_dtypes(exclude="number").columns.tolist()
    prepr=build_preprocessor(num_cols,cat_cols)
    model=Ridge(alpha=1.0)
    X_tr,X_te,y_tr,y_te=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    pipe=Pipeline([("prep",prepr),("reg",model)]); pipe.fit(X_tr,y_tr)
    st.metric("Ridge R²",f"{pipe.score(X_te,y_te):.3f}")

st.caption("v3 – NaN‑safe")
