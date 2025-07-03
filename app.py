
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import urllib.error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.signal import savgol_filter
from mlxtend.frequent_patterns import apriori, association_rules
import base64, warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG -----------------
DEFAULT_URL = "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Analytics Dashboard", layout="wide")

# ------------- Data Loader ----------------
@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except (urllib.error.HTTPError, urllib.error.URLError, FileNotFoundError) as e:
        st.error(f"❌ Unable to load CSV: {e}")
        st.stop()

url = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
uploaded = st.sidebar.file_uploader("Or upload CSV", type="csv")
df = load_csv(uploaded) if uploaded else load_csv(url)
st.sidebar.success(f"{len(df):,} rows × {df.shape[1]} cols loaded")

# ---------- Feature Engineering ----------
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1,"No":0,"Undecided":0})

# ---------- Helper functions -------------
def make_preprocessor(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for col in ["Lead_ID","Company_Name"]:
        if col in cat: cat = cat.drop(col)
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("sc",StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def download_link(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

# ---------- Tabs ----------
tab_titles = ["📊 Visualisation","🎯 Classification","🔀 Clustering","🔗 Assoc Rules","📈 Regression"]
tabs = st.tabs(tab_titles)

# =====================================================
# 1. Visualisation
# =====================================================
with tabs[0]:
    st.header("Data Visualisation – 10 Key Insights")

    # 1 Funnel by Sales_Stage
    if "Sales_Stage" in df.columns:
        stage_counts = df["Sales_Stage"].value_counts()
        fig = go.Figure(go.Funnel(y=stage_counts.index, x=stage_counts.values,
                                  textinfo="value+percent initial"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 1.** Sales funnel showing drop‑off at each stage.")

    # 2 Conversion by Lead Source
    if {"Lead_Source","Is_Converted"}.issubset(df.columns):
        conv_src = pd.crosstab(df["Lead_Source"], df["Is_Converted"], normalize="index")*100
        fig = px.bar(conv_src, barmode="stack", title="Conversion by Lead Source", labels={"value":"%"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 2.** Website leads convert at highest rate.")

    # 3 Region vs Product Category heatmap
    if {"Region","Product_Category"}.issubset(df.columns):
        heat = pd.crosstab(df["Region"], df["Product_Category"])
        st.plotly_chart(px.imshow(heat, text_auto=True, title="Demand heat‑map"), use_container_width=True)
        st.caption("**Fig 3.** Category demand differs by region.")

    # 4 Monthly Spend distribution
    if "Monthly_Spend_Parts_INR" in df.columns:
        st.plotly_chart(px.box(df, y="Monthly_Spend_Parts_INR",
                               title="Monthly Spend Distribution"), use_container_width=True)
        st.caption("**Fig 4.** Spend varies widely across customers.")

    # 5 Expected vs Current spend uplift
    if {"Expected_Monthly_Spend_NewVendor_INR","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        uplift = (df["Expected_Monthly_Spend_NewVendor_INR"] / df["Monthly_Spend_Parts_INR"] - 1)*100
        st.plotly_chart(px.histogram(uplift, nbins=30,
                                     title="Projected Revenue Uplift (%)"), use_container_width=True)
        st.caption("**Fig 5.** Many leads promise >10 % revenue uplift.")

    # 6 Delivery importance vs conversion
    if {"Delivery_Speed_Importance","Is_Converted"}.issubset(df.columns):
        conv = df.groupby("Delivery_Speed_Importance")["Is_Converted"].mean()*100
        st.plotly_chart(px.line(conv, markers=True,
                                 title="Conversion vs Delivery Priority"), use_container_width=True)
        st.caption("**Fig 6.** Higher delivery priority correlates with conversion.")

    # 7 CRM engagement distribution
    if "CRM_Engagement_Score" in df.columns:
        st.plotly_chart(px.histogram(df, x="CRM_Engagement_Score",
                                     nbins=10, title="CRM Engagement Score"), use_container_width=True)
        st.caption("**Fig 7.** Engagement skewed toward mid-high scores.")

    # 8 Correlation matrix
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 3:
        corr = df[num_cols].corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Numeric Correlations"), use_container_width=True)
        st.caption("**Fig 8.** Key numeric correlations.")

    # 9 Order frequency bars
    if "Average_Order_Frequency" in df.columns:
        st.plotly_chart(px.bar(df["Average_Order_Frequency"].value_counts(),
                               title="Order Frequency Distribution"), use_container_width=True)
        st.caption("**Fig 9.** Weekly orders dominate.")

    # 10 Years vs Spend with Savitzky-Golay trend
    if {"Years_In_Operation","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        scatter = df[["Years_In_Operation","Monthly_Spend_Parts_INR"]].dropna().sort_values("Years_In_Operation")
        fig = px.scatter(df, x="Years_In_Operation", y="Monthly_Spend_Parts_INR",
                         title="Maturity vs Spend")
        if len(scatter) >= 11:
            smooth = savgol_filter(scatter["Monthly_Spend_Parts_INR"], 11, 2)
            fig.add_scatter(x=scatter["Years_In_Operation"], y=smooth,
                            mode="lines", line=dict(color="white", width=2),
                            name="Trend (Savgol)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 10.** Older businesses tend to spend more.")

# =====================================================
# 2. Classification
# =====================================================
with tabs[1]:
    st.header("Classification")
    target_opts = [c for c in ["Will_Trial_Bin","Is_Converted"] if c in df_proc.columns]
    target = st.selectbox("Target label", target_opts)
    if target:
        X = df_proc.drop(columns=[target]); y = df_proc[target]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                        stratify=y,random_state=42)
        prep = make_preprocessor(X)
        learners = {"KNN":KNeighborsClassifier(),
                    "Decision Tree":DecisionTreeClassifier(random_state=42),
                    "Random Forest":RandomForestClassifier(random_state=42),
                    "GBRT":GradientBoostingClassifier(random_state=42)}
        results=[]; proba_dict={}
        for name, model in learners.items():
            pipe = Pipeline([("prep",prep),("est",model)]).fit(X_train,y_train)
            y_pred = pipe.predict(X_test)
            if hasattr(pipe,"predict_proba"):
                proba_dict[name]=pipe.predict_proba(X_test)[:,1]
            rep = classification_report(y_test,y_pred,output_dict=True,zero_division=0)
            results.append({"Model":name,
                            "Accuracy":round(rep["accuracy"],3),
                            "Precision":round(rep["weighted avg"]["precision"],3),
                            "Recall":round(rep["weighted avg"]["recall"],3),
                            "F1":round(rep["weighted avg"]["f1-score"],3)})
        res_df = pd.DataFrame(results).set_index("Model")
        st.dataframe(res_df)

        # Confusion matrix
        cm_model = st.selectbox("Confusion matrix model", list(learners.keys()))
        if st.checkbox("Show confusion matrix"):
            pipe = Pipeline([("prep",prep),("est",learners[cm_model])]).fit(X_train,y_train)
            cm = confusion_matrix(y_test, pipe.predict(X_test))
            st.plotly_chart(px.imshow(cm, text_auto=True,
                                      labels=dict(x="Pred",y="True"),
                                      title=f"Confusion Matrix – {cm_model}"),
                            use_container_width=True)

        # ROC curves
        if st.checkbox("Show ROC curves"):
            fig = go.Figure()
            for name, prob in proba_dict.items():
                fpr,tpr,_ = roc_curve(y_test, prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC {auc(fpr,tpr):.2f})"))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines",
                                     line=dict(dash="dash"), showlegend=False))
            fig.update_layout(title="ROC Curves",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        # Batch prediction
        new_file = st.file_uploader("Upload data for prediction", type="csv", key="pred")
        if new_file:
            new_df = pd.read_csv(new_file)
            best = res_df["F1"].idxmax()
            st.info(f"Predicting with best model: **{best}**")
            best_pipe = Pipeline([("prep",prep),("est",learners[best])]).fit(X,y)
            new_df["prediction"] = best_pipe.predict(new_df)
            st.markdown(download_link(new_df, "predictions.csv"), unsafe_allow_html=True)

# =====================================================
# 3. Clustering
# =====================================================
with tabs[2]:
    st.header("Clustering (K‑Means)")
    k = st.slider("Number of clusters (k)", 2, 10, 4)
    num_cols = df_proc.select_dtypes(include="number").columns
    scaler = Pipeline([("imp",SimpleImputer(strategy="median")),
                       ("sc",StandardScaler())])
    X_scaled = scaler.fit_transform(df_proc[num_cols])
    inertia=[]
    for kk in range(2,11):
        inertia.append(KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X_scaled).inertia_)
    st.plotly_chart(px.line(x=list(range(2,11)), y=inertia,
                            markers=True,title="Elbow Chart"), use_container_width=True)
    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X_scaled)
    df_clust = df_proc.copy()
    df_clust["cluster"] = km.labels_
    persona = df_clust.groupby("cluster").agg({
        "Monthly_Spend_Parts_INR":"mean",
        "Expected_Monthly_Spend_NewVendor_INR":"mean",
        "Delivery_Speed_Importance":"mean",
        "Quality_Certification_Priority":"mean",
        "CRM_Engagement_Score":"mean"}).round(1)
    st.dataframe(persona)
    st.markdown(download_link(df_clust, "clustered_data.csv"), unsafe_allow_html=True)

# =====================================================
# 4. Association Rules
# =====================================================
with tabs[3]:
    st.header("Association Rule Mining")
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols) >= 2:
        col1 = st.selectbox("Column A", cat_cols)
        col2 = st.selectbox("Column B", [c for c in cat_cols if c != col1])
        sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        conf = st.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)
        if st.button("Run Apriori"):
            basket = pd.get_dummies(df[[col1,col2]])
            freq = apriori(basket, min_support=sup, use_colnames=True)
            if freq.empty:
                st.warning("No itemsets meet support threshold.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=conf)
                if rules.empty:
                    st.warning("No rules meet confidence threshold.")
                else:
                    st.dataframe(rules.sort_values("lift", ascending=False)
                                 .head(10)[["antecedents","consequents","support","confidence","lift"]],
                                 use_container_width=True)
    else:
        st.info("Need ≥2 categorical columns.")

# =====================================================
# 5. Regression
# =====================================================
with tabs[4]:
    st.header("Regression Models")
    targets = [c for c in ["Expected_Monthly_Spend_NewVendor_INR",
                           "Monthly_Spend_Parts_INR"] if c in df_proc.columns]
    target_r = st.selectbox("Numeric target", targets)
    if target_r:
        Xr = df_proc.drop(columns=[target_r]); yr = df_proc[target_r]
        prep_r = make_preprocessor(Xr)
        regs = {"Linear":LinearRegression(),
                "Ridge":Ridge(alpha=1.0),
                "Lasso":Lasso(alpha=0.1),
                "Decision Tree":DecisionTreeRegressor(random_state=42)}
        rows=[]
        Xtr,Xts,ytr,yts = train_test_split(Xr,yr,test_size=0.2,random_state=42)
        for name, reg in regs.items():
            pipe=Pipeline([("prep",prep_r),("reg",reg)]).fit(Xtr,ytr)
            pred = pipe.predict(Xts)
            rows.append({"Model":name,
                         "R²":round(r2_score(yts,pred),3),
                         "MAE":round(mean_absolute_error(yts,pred),1),
                         "RMSE":round(np.sqrt(mean_squared_error(yts,pred)),1)})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))
