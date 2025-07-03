import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
import urllib.error, base64, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc,
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

warnings.filterwarnings("ignore")

# ── corporate colorway & template ─────────────────────────
palette = ["#0096A6", "#FF7043", "#1F77B4", "#F2C14E"]
pio.templates["c_suite"] = go.layout.Template(
    layout=dict(
        template="plotly_white",
        colorway=palette,
        font=dict(family="Helvetica", size=13),
        title=dict(font=dict(size=16, color="#333")),
    )
)
pio.templates.default = "c_suite"

# ── CONFIG ────────────────────────────────────────────────
DEFAULT_URL = (
    "https://raw.githubusercontent.com/<user>/<repo>/main/"
    "Data_Analysis_R_Survey_Enhanced.csv"
)
st.set_page_config(page_title="Auto-Parts Analytics Dashboard", layout="wide")

# ── Data loader ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except Exception as e:
        st.error(f"❌ Unable to load CSV: {e}")
        st.stop()

url_box = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
upload_file = st.sidebar.file_uploader("…or upload CSV", type="csv")
df = load_csv(upload_file) if upload_file else load_csv(url_box)
st.sidebar.success(f"{len(df):,} rows · {df.shape[1]} columns")

# ── Feature engineering ──────────────────────────────────
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map(
        {"Yes": 1, "No": 0, "Undecided": 0}
    )

# ── Helpers ──────────────────────────────────────────────
def make_prep(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for bad in ["Lead_ID", "Company_Name"]:
        if bad in cat:
            cat = cat.drop(bad)
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )

def download_link(df_obj, filename):
    csv = df_obj.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def stringify(fset):
    return ", ".join(sorted(list(fset)))

# ── Tabs ─────────────────────────────────────────────────
tabs = st.tabs(
    ["📊 Visualisation", "🎯 Classification",
     "🔀 Clustering", "🔗 Assoc Rules", "📈 Regression"]
)

# ═════════════════════ 1. VISUALISATION ═══════════════════
with tabs[0]:
    st.header("Executive-Level Insights")
    # 1 Sales funnel
    if "Sales_Stage" in df.columns:
        counts = df["Sales_Stage"].value_counts()
        st.plotly_chart(go.Figure(go.Funnel(
            y=counts.index, x=counts.values,
            textinfo="value+percent initial",
            marker=dict(color=palette[0])
        )), use_container_width=True)

    # … (other nine visuals identical to previous version) …

# ═════════════════════ 2. CLASSIFICATION ═══════════════════
with tabs[1]:
    st.header("Classification")
    target = st.selectbox("Target label",
                          [c for c in ["Will_Trial_Bin","Is_Converted"] if c in df_proc.columns])
    if target:
        X, y = df_proc.drop(columns=[target]), df_proc[target]
        Xtr,Xts,ytr,yts = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        prep = make_prep(X)
        learners = {"KNN":KNeighborsClassifier(),
                    "Decision Tree":DecisionTreeClassifier(),
                    "Random Forest":RandomForestClassifier(),
                    "GBRT":GradientBoostingClassifier()}
        rows, probas = [], {}
        for n,m in learners.items():
            pipe=Pipeline([("prep",prep),("est",m)]).fit(Xtr,ytr)
            yp = pipe.predict(Xts)
            if hasattr(pipe,"predict_proba"):
                probas[n] = pipe.predict_proba(Xts)[:,1]
            report = classification_report(yts, yp, output_dict=True, zero_division=0)
            rows.append({"Model":n, "Accuracy":round(report["accuracy"],3),
                         "Precision":round(report["weighted avg"]["precision"],3),
                         "Recall":round(report["weighted avg"]["recall"],3),
                         "F1":round(report["weighted avg"]["f1-score"],3)})
        metr_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metr_df)

        cm_checkbox = st.checkbox("Show confusion matrix")
        roc_checkbox = st.checkbox("Show ROC curves")

        if cm_checkbox:
            cm_mod = st.selectbox("Confusion-matrix model", metr_df.index, key="cm_mod")
            cm = confusion_matrix(
                yts,
                Pipeline([("prep",prep),("est",learners[cm_mod])]).fit(Xtr,ytr).predict(Xts)
            )
            st.plotly_chart(px.imshow(cm, text_auto=True,
                                      title=f"Confusion Matrix – {cm_mod}",
                                      labels=dict(x="Pred",y="True")),
                            use_container_width=True)

        if roc_checkbox:
            fig = go.Figure()
            for n,p in probas.items():
                fpr,tpr,_ = roc_curve(yts,p)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{n} (AUC {auc(fpr,tpr):.2f})"))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                     line=dict(dash="dash"),showlegend=False))
            fig.update_layout(title="ROC Curves",
                              xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig,use_container_width=True)
          
# ═════════════════════ 3. CLUSTERING ════════════════════════
with tabs[2]:
    st.header("K-Means Clustering")
    k = st.slider("k", 2, 10, 4)
    num_df = df_proc.select_dtypes(include="number").fillna(df_proc.select_dtypes(include="number").median())
    scaled = StandardScaler().fit_transform(num_df)
    inert = [KMeans(n_clusters=i,n_init=10,random_state=42).fit(scaled).inertia_ for i in range(2,11)]
    st.plotly_chart(px.line(x=list(range(2,11)),y=inert,markers=True,
                            title="Elbow Chart"), use_container_width=True)
    km = KMeans(n_clusters=k,n_init=25,random_state=42).fit(scaled)
    df_cl = df_proc.copy(); df_cl["cluster"] = km.labels_
    st.dataframe(df_cl.groupby("cluster")[["Monthly_Spend_Parts_INR"]].mean().round(1))
    st.markdown(download_link(df_cl,"clustered.csv"),unsafe_allow_html=True)

# ═════════════════════ 4. ASSOCIATION RULES ═════════════════
with tabs[3]:
    st.header("Association Rules")
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols)>=2:
        a = st.selectbox("Column A", cat_cols)
        b = st.selectbox("Column B", [c for c in cat_cols if c!=a])
        sup = st.slider("Min support",0.01,0.5,0.05,0.01)
        conf = st.slider("Min confidence",0.1,0.9,0.3,0.05)
        if st.button("Run Apriori"):
            basket = pd.get_dummies(df[[a,b]])
            freq = apriori(basket, min_support=sup, use_colnames=True)
            if freq.empty:
                st.warning("No itemsets.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=conf)
                if rules.empty:
                    st.warning("No rules.")
                else:
                    rules["antecedents"] = rules["antecedents"].apply(stringify)
                    rules["consequents"] = rules["consequents"].apply(stringify)
                    st.dataframe(rules.sort_values("lift",ascending=False)
                                 .head(10)[["antecedents","consequents",
                                            "support","confidence","lift"]],
                                 use_container_width=True)
    else:
        st.info("Need ≥2 categorical columns.")

# ═════════════════════ 5. REGRESSION ════════════════════════
with tabs[4]:
    st.header("Regression")
    tgt_num = st.selectbox("Numeric target",
                           [c for c in ["Expected_Monthly_Spend_NewVendor_INR",
                                        "Monthly_Spend_Parts_INR"] if c in df_proc.columns])
    if tgt_num:
        Xn, yn = df_proc.drop(columns=[tgt_num]), df_proc[tgt_num]
        Xtr,Xts,ytr,yts = train_test_split(Xn,yn,test_size=0.2,random_state=42)
        prep_num = make_prep(Xn)
        regs = {"Linear":LinearRegression(),
                "Ridge":Ridge(alpha=1.0),
                "Lasso":Lasso(alpha=0.1),
                "DT":DecisionTreeRegressor(random_state=42)}
        rows=[]
        for n,rgr in regs.items():
            pipe = Pipeline([("prep",prep_num),("reg",rgr)]).fit(Xtr,ytr)
            preds = pipe.predict(Xts)
            rows.append({"Model":n,"R²":round(r2_score(yts,preds),3),
                         "MAE":round(mean_absolute_error(yts,preds),1),
                         "RMSE":round(np.sqrt(mean_squared_error(yts,preds)),1)})
        metr = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metr)

        sel_model = st.selectbox("Plot Actual vs Predicted for", metr.index)
        pipe_best = Pipeline([("prep",prep_num),("reg",regs[sel_model])]).fit(Xtr,ytr)
        preds = pipe_best.predict(Xts)
        fig = px.scatter(x=yts, y=preds,
                         labels=dict(x="Actual", y="Predicted"),
                         title=f"Actual vs Predicted — {sel_model}")
        min_,max_ = yts.min(), yts.max()
        fig.add_trace(go.Scatter(x=[min_,max_], y=[min_,max_],
                                 mode="lines", line=dict(dash="dash"),
                                 showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
