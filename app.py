
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
import plotly.io as pio, warnings
warnings.filterwarnings('ignore')
pio.templates.default = "plotly_dark"

DEFAULT_URL = "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Analytics Dashboard", layout="wide")

@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

url = st.sidebar.text_input("Raw CSV URL", value=DEFAULT_URL)
up = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_csv(up) if up else load_csv(url)
st.sidebar.success(f"{len(df):,} rows, {df.shape[1]} columns")

df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1,"No":0,"Undecided":0})

def make_prep(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for c in ["Lead_ID","Company_Name"]:
        if c in cat: cat = cat.drop(c)
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("sc",StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def dl_link(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def stringify(fset):
    return ", ".join(sorted(list(fset)))

tabs = st.tabs(["📊 Visualisation","🎯 Classification","🔀 Clustering","🔗 Assoc Rules","📈 Regression"])

# ===== VIS =====
with tabs[0]:
    st.header("Interactive Descriptive Insights")
    if "Sales_Stage" in df.columns:
        counts = df["Sales_Stage"].value_counts()
        st.plotly_chart(go.Figure(go.Funnel(y=counts.index,x=counts.values,textinfo="value+percent initial")),
                        use_container_width=True)
    if {"Years_In_Operation","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        scatter = df[["Years_In_Operation","Monthly_Spend_Parts_INR"]].dropna().sort_values("Years_In_Operation")
        fig = px.scatter(df, x="Years_In_Operation", y="Monthly_Spend_Parts_INR",
                         color="Business_Type" if "Business_Type" in df.columns else None,
                         title="Maturity vs Spend")
        if len(scatter)>=11:
            smooth = savgol_filter(scatter["Monthly_Spend_Parts_INR"], 11, 2)
            fig.add_scatter(x=scatter["Years_In_Operation"], y=smooth,
                            mode="lines", line=dict(color="white"), name="Trend")
        st.plotly_chart(fig, use_container_width=True)

# ===== CLASS =====
with tabs[1]:
    st.header("Classification")
    target = st.selectbox("Target", [c for c in ["Will_Trial_Bin","Is_Converted"] if c in df_proc.columns])
    if target:
        X=df_proc.drop(columns=[target]); y=df_proc[target]
        Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        prep = make_prep(X)
        learners={"KNN":KNeighborsClassifier(), "DT":DecisionTreeClassifier(),
                  "RF":RandomForestClassifier(), "GBRT":GradientBoostingClassifier()}
        rows=[]; probas={}
        for n,m in learners.items():
            pipe=Pipeline([("prep",prep),("est",m)]).fit(Xtr,ytr)
            y_pred=pipe.predict(Xts)
            if hasattr(pipe,"predict_proba"):
                probas[n]=pipe.predict_proba(Xts)[:,1]
            rep=classification_report(yts,y_pred,output_dict=True,zero_division=0)
            rows.append({"Model":n,"Accuracy":round(rep["accuracy"],3),
                         "Precision":round(rep["weighted avg"]["precision"],3),
                         "Recall":round(rep["weighted avg"]["recall"],3),
                         "F1":round(rep["weighted avg"]["f1-score"],3)})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))
        cm_mod = st.selectbox("Confusion Matrix model", list(learners.keys()))
        if st.button("Show CM"):
            pipe=Pipeline([("prep",prep),("est",learners[cm_mod])]).fit(Xtr,ytr)
            cm=confusion_matrix(yts,pipe.predict(Xts))
            st.plotly_chart(px.imshow(cm,text_auto=True,labels=dict(x="Pred",y="True"),
                                      title=f"CM – {cm_mod}"),use_container_width=True)
        if st.checkbox("ROC curves"):
            fig=go.Figure()
            for n,p in probas.items():
                fpr,tpr,_=roc_curve(yts,p); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                                                                      name=f"{n} (AUC {auc(fpr,tpr):.2f})"))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash"),showlegend=False))
            st.plotly_chart(fig,use_container_width=True)

# ===== CLUSTER =====
with tabs[2]:
    st.header("K‑Means Clustering")
    k=st.slider("k",2,10,4)
    num=df_proc.select_dtypes(include="number").columns
    scaled=StandardScaler().fit_transform(df_proc[num])
    inert=[KMeans(n_clusters=i,n_init=10,random_state=42).fit(scaled).inertia_ for i in range(2,11)]
    st.plotly_chart(px.line(x=list(range(2,11)),y=inert,markers=True,title="Elbow"),use_container_width=True)
    km=KMeans(n_clusters=k,n_init=25,random_state=42).fit(scaled)
    df_cl=df_proc.copy(); df_cl["cluster"]=km.labels_
    st.dataframe(df_cl.groupby("cluster")[["Monthly_Spend_Parts_INR"]].mean().round(1))
    st.markdown(dl_link(df_cl,"clusters.csv"),unsafe_allow_html=True)

# ===== ASSOC =====
with tabs[3]:
    st.header("Association Rules")
    cat=df.select_dtypes(exclude="number").columns
    if len(cat)>=2:
        a=st.selectbox("Column A",cat)
        b=st.selectbox("Column B",[c for c in cat if c!=a])
        sup=st.slider("Min support",0.01,0.5,0.05,0.01)
        conf=st.slider("Min confidence",0.1,0.9,0.3,0.05)
        if st.button("Run Apriori"):
            bask=pd.get_dummies(df[[a,b]])
            freq=apriori(bask,min_support=sup,use_colnames=True)
            if freq.empty: st.warning("No itemsets."); 
            else:
                rules=association_rules(freq,metric="confidence",min_threshold=conf)
                if rules.empty: st.warning("No rules.");
                else:
                    rules["antecedents"]=rules["antecedents"].apply(stringify)
                    rules["consequents"]=rules["consequents"].apply(stringify)
                    st.dataframe(rules.sort_values("lift",ascending=False)
                                 .head(10)[["antecedents","consequents","support","confidence","lift"]],
                                 use_container_width=True)
    else:
        st.info("Need categorical columns.")

# ===== REGRESSION =====
with tabs[4]:
    st.header("Regression")
    tgt=st.selectbox("Numeric target",[c for c in ["Expected_Monthly_Spend_NewVendor_INR","Monthly_Spend_Parts_INR"] if c in df_proc.columns])
    if tgt:
        X=df_proc.drop(columns=[tgt]); y=df_proc[tgt]
        Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.2,random_state=42)
        prep=make_prep(X)
        regs={"Linear":LinearRegression(),"Ridge":Ridge(alpha=1.0),
              "Lasso":Lasso(alpha=0.1),"DT":DecisionTreeRegressor(random_state=42)}
        rows=[]
        for n,r in regs.items():
            pipe=Pipeline([("prep",prep),("reg",r)]).fit(Xtr,ytr)
            pred=pipe.predict(Xts)
            rows.append({"Model":n,"R²":round(r2_score(yts,pred),3),
                         "MAE":round(mean_absolute_error(yts,pred),1),
                         "RMSE":round(np.sqrt(mean_squared_error(yts,pred)),1)})
        res=pd.DataFrame(rows).set_index("Model")
        st.dataframe(res)
        model_plot=st.selectbox("Plot for model",res.index)
        best_pipe=Pipeline([("prep",prep),("reg",regs[model_plot])]).fit(Xtr,ytr)
        pred=best_pipe.predict(Xts)
        fig=px.scatter(x=yts,y=pred,labels=dict(x="Actual",y="Predicted"),
                       title=f"Actual vs Predicted – {model_plot}")
        min_,max_=yts.min(),yts.max()
        fig.add_trace(go.Scatter(x=[min_,max_],y=[min_,max_],mode="lines",line=dict(dash="dash"),showlegend=False))
        st.plotly_chart(fig,use_container_width=True)
