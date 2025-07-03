
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
from mlxtend.frequent_patterns import apriori, association_rules
import base64, io

# -------------- EDIT THIS LINE -----------------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/Data_Analysis_R_Survey_Enhanced.csv"
# -----------------------------------------------------------------

st.set_page_config(page_title="Auto‑Parts Lead Analytics", layout="wide")

@st.cache_data(ttl=3600)
def load_csv(src):
    return pd.read_csv(src)

# Sidebar data source
st.sidebar.header("Data source")

url_input = st.sidebar.text_input("GitHub raw CSV URL (optional)", value=DEFAULT_URL)
uploaded_file = st.sidebar.file_uploader("…or upload local CSV", type="csv")

# Decide which source to load
if uploaded_file is not None:
    df = load_csv(uploaded_file)
elif url_input:
    try:
        df = load_csv(url_input)
    except Exception as e:
        st.error(f"Unable to read CSV from URL. Details: {e}")
        st.stop()
else:
    st.error("Please provide a CSV URL or upload a file.")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows × {df.shape[1]} cols")

# Simple preprocessing helpers
binary_map = {"Yes":1, "No":0, "Undecided":0}
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map(binary_map)

num_cols = df_proc.select_dtypes(include="number").columns.tolist()
cat_cols = df_proc.select_dtypes(exclude="number").columns.tolist()
for c in ["Lead_ID", "Company_Name"]:
    if c in cat_cols:
        cat_cols.remove(c)

# Tabs
tab_titles = ["📊 Visualisation", "🎯 Classification", "🔀 Clustering", "🔗 Assoc Rules", "📈 Regression"]
tabs = st.tabs(tab_titles)

# 1 Visualisation
with tabs[0]:
    st.header("Descriptive Visualisations")
    with st.sidebar.expander("Filters", expanded=False):
        region_sel = st.multiselect("Region", options=sorted(df["Region"].unique()) if "Region" in df.columns else [])
        source_sel = st.multiselect("Lead Source", options=sorted(df["Lead_Source"].unique()) if "Lead_Source" in df.columns else [])
    dff = df.copy()
    if region_sel:
        dff = dff[dff["Region"].isin(region_sel)]
    if source_sel:
        dff = dff[dff["Lead_Source"].isin(source_sel)]
    # example charts
    if "Is_Converted" in dff.columns:
        conv = dff["Is_Converted"].value_counts(normalize=True)*100
        st.plotly_chart(px.pie(conv, values=conv.values, names=conv.index,
                                title="Lead Conversion Split"), use_container_width=True)
    if {"Region","Is_Converted"}.issubset(dff.columns):
        st.plotly_chart(px.bar(dff, x="Region", color="Is_Converted", barmode="group",
                               title="Conversion by Region"), use_container_width=True)

# 2 Classification
with tabs[1]:
    st.header("Binary Classification")
    target = st.selectbox("Target label", options=["Will_Trial_Bin","Is_Converted"])
    if target not in df_proc.columns:
        st.warning("Chosen target not found.")
    else:
        X = df_proc.drop(columns=[target])
        y = df_proc[target]
        num = X.select_dtypes(include="number").columns
        cat = X.select_dtypes(exclude="number").columns
        prep = ColumnTransformer([("num", StandardScaler(), num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        models = {"KNN":KNeighborsClassifier(),
                  "DT":DecisionTreeClassifier(random_state=42),
                  "RF":RandomForestClassifier(random_state=42),
                  "GBRT":GradientBoostingClassifier(random_state=42)}
        test_pct = st.slider("Test %",10,40,20,5)
        X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=test_pct/100,
                                                  stratify=y, random_state=42)
        results = {}
        for n,m in models.items():
            pipe = Pipeline([("pre",prep),("m",m)])
            pipe.fit(X_tr,y_tr)
            y_pred = pipe.predict(X_te)
            rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
            results[n] = rep["weighted avg"]; results[n]["accuracy"]=rep["accuracy"]
        res_df = (pd.DataFrame(results).T
                  .loc[:,["accuracy","precision","recall","f1-score"]]
                  .round(3).sort_values("f1-score",ascending=False))
        st.dataframe(res_df,use_container_width=True)
        # confusion matrix toggle
        cm_mod = st.selectbox("Model for CM", options=list(models.keys()))
        if st.button("Show CM"):
            pipe = Pipeline([("pre",prep),("m",models[cm_mod])])
            pipe.fit(X_tr,y_tr); cm = confusion_matrix(y_te, pipe.predict(X_te))
            st.plotly_chart(px.imshow(cm,text_auto=True,
                                      x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                                      title=f"Confusion Matrix – {cm_mod}"),
                            use_container_width=True)
        if st.checkbox("ROC curves"):
            plt.figure()
            for n,m in models.items():
                pipe=Pipeline([("pre",prep),("m",m)]); pipe.fit(X_tr,y_tr)
                if hasattr(pipe,"predict_proba"):
                    prob = pipe.predict_proba(X_te)[:,1]
                    fpr,tpr,_=roc_curve(y_te,prob); plt.plot(fpr,tpr,label=n)
            plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.legend(); st.pyplot(plt.gcf())

# 3 Clustering
with tabs[2]:
    st.header("K‑Means Clustering")
    k = st.slider("k",2,10,4)
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(df_proc.select_dtypes(include="number"))
    inert=[]
    for k_i in range(2,11):
        inert.append(KMeans(n_clusters=k_i,n_init=10,random_state=42).fit(X_scale).inertia_)
    st.plotly_chart(px.line(x=list(range(2,11)),y=inert,markers=True,title="Elbow"),use_container_width=True)
    km = KMeans(n_clusters=k,n_init=25,random_state=42).fit(X_scale)
    df_proc["cluster"]=km.labels_
    st.dataframe(df_proc.groupby("cluster").mean().round(1),use_container_width=True)
    st.download_button("Download clusters", df_proc.to_csv(index=False).encode(),
                       file_name="clustered.csv", mime="text/csv")

# 4 Assoc rules
with tabs[3]:
    st.header("Association Rules")
    cats = df_proc.select_dtypes(exclude="number").columns.tolist()
    if len(cats)<2:
        st.info("Need at least two categorical columns.")
    else:
        c1 = st.selectbox("Column 1", cats)
        c2 = st.selectbox("Column 2", [c for c in cats if c!=c1])
        min_sup = st.number_input("Min support",0.01,1.0,0.05,0.01)
        min_conf = st.number_input("Min confidence",0.1,1.0,0.3,0.05)
        basket = pd.get_dummies(df[[c1,c2]])
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        if rules.empty:
            st.warning("No rules meet thresholds.")
        else:
            st.dataframe(rules.sort_values("lift",ascending=False)
                         .head(10)[["antecedents","consequents","support","confidence","lift"]],
                         use_container_width=True)

# 5 Regression
with tabs[4]:
    st.header("Regression")
    target_r = st.selectbox("Numeric target",
                            options=["Expected_Monthly_Spend_NewVendor_INR","Monthly_Spend_Parts_INR"])
    Xr = df_proc.drop(columns=[target_r]); yr = df_proc[target_r]
    num_r = Xr.select_dtypes(include="number").columns
    cat_r = Xr.select_dtypes(exclude="number").columns
    prep_r = ColumnTransformer([("num",StandardScaler(),num_r),
                                ("cat",OneHotEncoder(handle_unknown="ignore"),cat_r)])
    models_r = {"Linear":LinearRegression(),
                "Ridge":Ridge(alpha=1.0),
                "Lasso":Lasso(alpha=0.1)}
    for n,m in models_r.items():
        pipe=Pipeline([("pre",prep_r),("reg",m)])
        X_tr,X_te,y_tr,y_te = train_test_split(Xr,yr,test_size=0.2,random_state=42)
        pipe.fit(X_tr,y_tr); score=pipe.score(X_te,y_te)
        st.metric(f"{n} R²",f"{score:.3f}")

st.caption("© 2025 Auto‑Parts Analytics Dashboard")
