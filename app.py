
import streamlit as st
import pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go, urllib.error, base64
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
import warnings
warnings.filterwarnings('ignore')

DEFAULT_URL = "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Dashboard", layout="wide")

@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except (urllib.error.URLError, urllib.error.HTTPError, FileNotFoundError) as e:
        st.error(f"❌ {e}")
        st.stop()

url = st.sidebar.text_input("Raw CSV URL", value=DEFAULT_URL)
upl = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_csv(upl) if upl else load_csv(url)
st.sidebar.success(f"{len(df):,} rows loaded")

df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1,"No":0,"Undecided":0})

def prep(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for c in ["Lead_ID","Company_Name"]:
        if c in cat:
            cat = cat.drop(c)
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("sc",StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def dlink(df, name):
    b = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a download="{name}" href="data:file/csv;base64,{b}">Download CSV</a>'

tabs = st.tabs(["EDA","Classification","Clustering","Assoc Rules","Regression"])

# --- Association helper to pretty print itemsets ---
def stringify(itemset):
    return ", ".join(sorted(list(itemset)))

# ====== Assoc Rules tab will be implemented later =====

# For brevity include only the requested modifications (Assoc Rules and Regression scatter):
with tabs[3]:
    st.header("Association Rules")
    cats = df.select_dtypes(exclude="number").columns
    if len(cats) >= 2:
        colA = st.selectbox("Column A", cats)
        colB = st.selectbox("Column B", [c for c in cats if c!=colA])
        min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)
        if st.button("Generate rules"):
            basket = pd.get_dummies(df[[colA, colB]])
            freq = apriori(basket, min_support=min_sup, use_colnames=True)
            if freq.empty:
                st.warning("No itemsets at this support.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                if rules.empty:
                    st.warning("No rules meet confidence threshold.")
                else:
                    rules["antecedents"] = rules["antecedents"].apply(stringify)
                    rules["consequents"] = rules["consequents"].apply(stringify)
                    st.dataframe(rules.sort_values("lift", ascending=False)
                                 .head(10)[["antecedents","consequents","support","confidence","lift"]],
                                 use_container_width=True)
    else:
        st.info("Need at least two categorical columns.")

with tabs[4]:
    st.header("Regression")
    targets = [c for c in ["Expected_Monthly_Spend_NewVendor_INR","Monthly_Spend_Parts_INR"] if c in df_proc.columns]
    tgt = st.selectbox("Target", targets)
    if tgt:
        X = df_proc.drop(columns=[tgt]); y = df_proc[tgt]
        Xtr,Xts,ytr,yts = train_test_split(X,y,test_size=0.2,random_state=42)
        pre = prep(X)
        regs = {"Linear":LinearRegression(),
                "Ridge":Ridge(alpha=1.0),
                "Lasso":Lasso(alpha=0.1),
                "DT":DecisionTreeRegressor(random_state=42)}
        metrics = []
        for n,m in regs.items():
            pipe = Pipeline([("pre",pre),("reg",m)]).fit(Xtr,ytr)
            pred = pipe.predict(Xts)
            metrics.append({"Model":n,
                            "R²":round(r2_score(yts,pred),3),
                            "MAE":round(mean_absolute_error(yts,pred),1),
                            "RMSE":round(np.sqrt(mean_squared_error(yts,pred)),1)})
        metrics_df = pd.DataFrame(metrics).set_index("Model")
        st.dataframe(metrics_df)

        plot_model = st.selectbox("Plot predictions for model", metrics_df.index)
        pipe_sel = Pipeline([("pre",pre),("reg",regs[plot_model])]).fit(Xtr,ytr)
        preds = pipe_sel.predict(Xts)
        fig = px.scatter(x=yts, y=preds, labels={"x":"Actual","y":"Predicted"},
                         title=f"Actual vs Predicted – {plot_model}")
        line_min, line_max = yts.min(), yts.max()
        fig.add_trace(go.Scatter(x=[line_min,line_max], y=[line_min,line_max],
                                 mode="lines", line=dict(dash="dash"), showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
