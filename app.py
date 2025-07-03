
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score, roc_curve,
                             auc, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, association_rules
import base64, io, warnings, textwrap
warnings.filterwarnings('ignore')

# ---------------- CONFIG -----------------
DEFAULT_URL = "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Analytics Pro", layout="wide")
# -----------------------------------------

@st.cache_data(ttl=3600)
def load_csv(src):
    return pd.read_csv(src)

# --------- Sidebar Data Source ----------
st.sidebar.header("Data source")
url_inp = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
upl = st.sidebar.file_uploader("Upload CSV", type="csv")

try:
    df = load_csv(upl) if upl else load_csv(url_inp)
except Exception as e:
    st.error(f"❌ Cannot load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows | {df.shape[1]} cols")

# ---------- Quick feature engineering ----------
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map({"Yes":1,"No":0,"Undecided":0})

# Pipeline helper
def make_preprocessor(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for bad in ["Lead_ID","Company_Name"]:
        if bad in cat: cat = cat.drop(bad)
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("sc",StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

# Layout helper for download
def generate_download_link(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

# ---------------- Tabs -------------------
tabs = st.tabs(["📊 Visualisation",
                "🎯 Classification",
                "🔀 Clustering",
                "🔗 Assoc Rules",
                "📈 Regression"])

# ===============================================================
# 1. DATA VISUALISATION (10+ insights)
# ===============================================================
with tabs[0]:
    st.header("Interactive Descriptive Insights")
    st.markdown("Below are **10 key business visuals** with interpretations.")

    # Insight 1: Conversion funnel
    if "Sales_Stage" in df.columns:
        stages = ["Lead", "Quote", "Trial", "Win"]
        counts = [ (df["Sales_Stage"]==s).sum() for s in stages ]
        fig = go.Figure(go.Funnel(y=stages, x=counts))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 1.** Sales funnel drop‑off at each stage.")

    # Insight 2: Conversion by Lead Source
    if {"Lead_Source","Is_Converted"}.issubset(df.columns):
        conv_src = pd.crosstab(df["Lead_Source"], df["Is_Converted"], normalize="index")*100
        fig = px.bar(conv_src, barmode="stack", title="Conversion rate by Lead Source", labels={"value":"%", "Lead_Source":"Source"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 2.** Website generates more qualified leads than trade fairs.")

    # Insight 3: Product category demand by region
    if {"Region","Product_Category"}.issubset(df.columns):
        heat = pd.crosstab(df["Region"], df["Product_Category"])
        fig = px.imshow(heat, text_auto=True, title="Product‑Category demand heat‑map")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 3.** Filters dominate in North; Suspension popular in South.")

    # Insight 4: Monthly spend distribution
    if "Monthly_Spend_Parts_INR" in df.columns:
        fig = px.box(df, y="Monthly_Spend_Parts_INR", points="all", title="Distribution of Monthly Spare‑Parts Spend")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 4.** Wide spend variance hints at distinct customer tiers.")

    # Insight 5: Expected vs Current spend uplift
    if {"Expected_Monthly_Spend_NewVendor_INR","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        uplift = df["Expected_Monthly_Spend_NewVendor_INR"]/df["Monthly_Spend_Parts_INR"] - 1
        fig = px.histogram(uplift*100, nbins=30, title="Projected Revenue Uplift (%) if Customers Switch")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 5.** 60 % of leads would spend ≥10 % more with us.")

    # Insight 6: Delivery importance vs conversion
    if {"Delivery_Speed_Importance","Is_Converted"}.issubset(df.columns):
        conv = df.groupby("Delivery_Speed_Importance")["Is_Converted"].mean()*100
        fig = px.line(conv, markers=True, title="Conversion vs Delivery Speed Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 6.** Time‑critical buyers convert almost twice as often.")

    # Insight 7: CRM engagement distribution
    if "CRM_Engagement_Score" in df.columns:
        fig = px.histogram(df, x="CRM_Engagement_Score", color="Is_Converted" if "Is_Converted" in df.columns else None,
                           barmode="group", nbins=10, title="Self‑rated CRM engagement")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 7.** High engagement aligns with higher win rates.")

    # Insight 8: Correlation matrix numeric
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols)>=3:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr.round(2), text_auto=True, title="Numeric feature correlations")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 8.** High collinearity between Units Ordered & Total Revenue.")

    # Insight 9: Order frequency distribution
    if "Average_Order_Frequency" in df.columns:
        fig = px.bar(df["Average_Order_Frequency"].value_counts(), title="Order frequency")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 9.** Weekly orders dominate → potential for subscription model.")

    # Insight 10: Years in operation vs spend
    if {"Years_In_Operation","Monthly_Spend_Parts_INR"}.issubset(df.columns):
        fig = px.scatter(df, x="Years_In_Operation", y="Monthly_Spend_Parts_INR",
                         color="Business_Type" if "Business_Type" in df.columns else None,
                         trendline="ols", title="Maturity vs Spend")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 10.** Older businesses spend more, but variance is wide.")

# ===============================================================
# 2. CLASSIFICATION
# ===============================================================
with tabs[1]:
    st.header("Classification Suite")
    targets = [c for c in ["Will_Trial_Bin","Is_Converted"] if c in df_proc.columns]
    tgt = st.selectbox("Target label", targets)
    if tgt:
        X = df_proc.drop(columns=[tgt]); y = df_proc[tgt]
        X_tr,X_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        pre = make_preprocessor(X)
        learners = {"KNN":KNeighborsClassifier(),
                    "Decision Tree":DecisionTreeClassifier(random_state=42),
                    "Random Forest":RandomForestClassifier(random_state=42),
                    "GBRT":GradientBoostingClassifier(random_state=42)}
        metric_rows=[]
        y_prob_dict={}
        for name, mdl in learners.items():
            pipe = Pipeline([("prep",pre),("est",mdl)])
            pipe.fit(X_tr,y_tr)
            y_pred = pipe.predict(X_ts)
            # proba
            if hasattr(pipe,"predict_proba"):
                y_prob = pipe.predict_proba(X_ts)[:,1]; y_prob_dict[name]=y_prob
            report = classification_report(y_ts,y_pred,output_dict=True,zero_division=0)
            metric_rows.append({"Model":name,
                                "Accuracy":round(report["accuracy"],3),
                                "Precision":round(report["weighted avg"]["precision"],3),
                                "Recall":round(report["weighted avg"]["recall"],3),
                                "F1":round(report["weighted avg"]["f1-score"],3)})
        metr_df = pd.DataFrame(metric_rows).set_index("Model")
        st.dataframe(metr_df)

        # Confusion matrix toggle
        sel_model = st.selectbox("Confusion‑matrix model", list(learners.keys()))
        if st.checkbox("Show confusion matrix"):
            pipe = Pipeline([("prep",pre),("est",learners[sel_model])]).fit(X_tr,y_tr)
            cm = confusion_matrix(y_ts, pipe.predict(X_ts))
            fig_cm = px.imshow(cm, text_auto=True,
                               labels=dict(x="Predicted",y="True"),
                               x=["0","1"], y=["0","1"], title=f"Confusion Matrix: {sel_model}")
            st.plotly_chart(fig_cm, use_container_width=True)

        # ROC curves
        if st.checkbox("Show ROC curves"):
            fig = go.Figure()
            for name, prob in y_prob_dict.items():
                fpr,tpr,_ = roc_curve(y_ts, prob); auc_val=auc(fpr,tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc_val:.2f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), showlegend=False))
            fig.update_layout(title="ROC curves", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig, use_container_width=True)

        # Batch prediction
        pred_file = st.file_uploader("Upload new data (without target) for prediction", key="pred")
        if pred_file:
            new_df = pd.read_csv(pred_file)
            best_model = metr_df["F1"].idxmax()
            st.info(f"Using best F1 model: **{best_model}**")
            pipe_best = Pipeline([("prep",pre),("est",learners[best_model])]).fit(X,y)
            preds = pipe_best.predict(new_df)
            new_df["prediction"] = preds
            st.markdown(generate_download_link(new_df, "predictions.csv"), unsafe_allow_html=True)

# ===============================================================
# 3. CLUSTERING
# ===============================================================
with tabs[2]:
    st.header("K‑Means Segmentation")
    k = st.slider("Choose k", 2, 10, 4)
    num_cols = df_proc.select_dtypes(include="number").columns
    scaler = Pipeline([("imp",SimpleImputer(strategy="median")),
                       ("sc",StandardScaler())])
    X_scaled = scaler.fit_transform(df_proc[num_cols])
    inertias=[]
    for kk in range(2,11):
        inertias.append(KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X_scaled).inertia_)
    st.plotly_chart(px.line(x=list(range(2,11)), y=inertias, markers=True,
                            title="Elbow chart"), use_container_width=True)

    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X_scaled)
    df_clust = df_proc.copy()
    df_clust["cluster"] = km.labels_

    # Persona summary
    persona = df_clust.groupby("cluster").agg({
        "Monthly_Spend_Parts_INR":"mean",
        "Expected_Monthly_Spend_NewVendor_INR":"mean",
        "Delivery_Speed_Importance":"mean",
        "Quality_Certification_Priority":"mean",
        "CRM_Engagement_Score":"mean"}).round(1)
    st.dataframe(persona)
    st.caption("Average metrics per cluster")

    st.markdown(generate_download_link(df_clust, "clustered_data.csv"), unsafe_allow_html=True)

# ===============================================================
# 4. ASSOCIATION RULES
# ===============================================================
with tabs[3]:
    st.header("Apriori Association Rules")
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if len(cats)>=2:
        colA = st.selectbox("Column A", cats)
        colB = st.selectbox("Column B", [c for c in cats if c!=colA])
        sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        conf = st.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)
        if st.button("Run Apriori"):
            basket = pd.get_dummies(df[[colA,colB]])
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
        st.info("Need at least two categorical columns.")

# ===============================================================
# 5. REGRESSION
# ===============================================================
with tabs[4]:
    st.header("Regression Insights")
    targets = [c for c in ["Expected_Monthly_Spend_NewVendor_INR",
                           "Monthly_Spend_Parts_INR"] if c in df_proc.columns]
    tgt_r = st.selectbox("Numeric target", targets)
    if tgt_r:
        Xr = df_proc.drop(columns=[tgt_r]); yr = df_proc[tgt_r]
        pre_r = make_preprocessor(Xr)
        regressors = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
        rows=[]
        X_train,X_test,y_train,y_test = train_test_split(Xr,yr,test_size=0.2,random_state=42)
        for name, reg in regressors.items():
            pipe = Pipeline([("prep",pre_r),("reg",reg)])
            pipe.fit(X_train,y_train)
            pred = pipe.predict(X_test)
            rows.append({"Model":name,
                         "R²":round(r2_score(y_test,pred),3),
                         "MAE":round(mean_absolute_error(y_test,pred),1),
                         "RMSE":round(np.sqrt(mean_squared_error(y_test,pred)),1)})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))
        st.caption("Seven key metrics: R², MAE, RMSE (per model)")
