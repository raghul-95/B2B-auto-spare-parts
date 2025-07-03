import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import io, base64, requests, json, textwrap

st.set_page_config(page_title="Auto‑Parts Lead Analytics", layout="wide")

# ----------------------------------------------------------------------
# Data loader
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(src, sep=","):
    if src.startswith("http"):
        df = pd.read_csv(src, sep=sep)
    else:
        df = pd.read_csv(src, sep=sep)
    return df

st.sidebar.header("Data source")
default_url = st.secrets.get("GITHUB_RAW_URL", "")
data_url = st.sidebar.text_input("GitHub raw CSV URL", value=default_url)
uploaded = st.sidebar.file_uploader("…or upload local CSV", type="csv")

if uploaded:
    df = load_data(uploaded)
elif data_url:
    try:
        df = load_data(data_url)
    except Exception as e:
        st.error(f"Failed to load URL: {e}")
        st.stop()
else:
    st.warning("Provide a CSV URL or upload a file ➜")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} rows • {df.shape[1]} cols")

# ----------------------------------------------------------------------
# Common preprocessing helpers
# ----------------------------------------------------------------------
binary_map = {"Yes":1, "No":0, "Undecided":0}
df_processed = df.copy()
if "Willing_To_Trial_NewVendor" in df_processed.columns:
    df_processed["Will_Trial_Bin"] = df_processed["Willing_To_Trial_NewVendor"].map(binary_map)

numeric_cols = df_processed.select_dtypes(include="number").columns.tolist()
cat_cols = df_processed.select_dtypes(exclude="number").columns.tolist()

# remove obvious IDs if present
for c in ["Lead_ID", "Company_Name"]:
    if c in cat_cols:
        cat_cols.remove(c)

# ----------------------------------------------------------------------
# Tab navigation
# ----------------------------------------------------------------------
tab_titles = ["📊 Data Visualisation", "🎯 Classification", "🔀 Clustering",
              "🔗 Assoc Rules", "📈 Regression"]
tabs = st.tabs(tab_titles)

# =========================================================================
# 1. DATA VISUALISATION
# =========================================================================
with tabs[0]:
    st.header("Descriptive Insights")
    st.markdown("Below are ten starter visualisations. Adjust filters in the sidebar "
                "to refine what you see.")

    # Sidebar filters
    with st.sidebar.expander("Filters"):
        region_pick = st.multiselect("Region(s)", options=sorted(df["Region"].unique())
                                     if "Region" in df.columns else [], default=None)
        src_pick = st.multiselect("Lead Source(s)", options=sorted(df["Lead_Source"].unique())
                                  if "Lead_Source" in df.columns else [], default=None)
    dff = df.copy()
    if region_pick:
        dff = dff[dff["Region"].isin(region_pick)]
    if src_pick:
        dff = dff[dff["Lead_Source"].isin(src_pick)]

    # 10 charts
    charts = []
    if "Is_Converted" in dff.columns:
        conv = dff["Is_Converted"].value_counts(normalize=True)*100
        fig = px.pie(conv, values=conv.values, names=conv.index,
                     title="Lead Conversion Split (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 1.** Overall conversion ratio of leads.")

    if {"Region","Is_Converted"}.issubset(dff.columns):
        fig = px.bar(dff, x="Region", color="Is_Converted",
                     barmode="group", title="Conversion rate by Region")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 2.** Comparison of converted vs non‑converted leads by region.")

    if {"Lead_Source","Is_Converted"}.issubset(dff.columns):
        fig = px.histogram(dff, x="Lead_Source", color="Is_Converted",
                           title="Conversion by Lead Source", barnorm="percent")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 3.** Website vs trade‑fair efficiency visualised.")

    if {"Product_Category","Region"}.issubset(dff.columns):
        pc = pd.crosstab(dff["Region"], dff["Product_Category"])
        fig = px.imshow(pc, aspect="auto", text_auto=True,
                        title="Product‑Category Demand by Region")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 4.** Heat‑map showing most‑demanded categories per region.")

    if {"Monthly_Spend_Parts_INR","Expected_Monthly_Spend_NewVendor_INR","Business_Type"}.issubset(dff.columns):
        melted = dff.melt(id_vars=["Business_Type"],
                          value_vars=["Monthly_Spend_Parts_INR",
                                      "Expected_Monthly_Spend_NewVendor_INR"],
                          var_name="Spend_Type", value_name="INR")
        fig = px.box(melted, x="Business_Type", y="INR",
                     color="Spend_Type",
                     title="Current vs Expected Spend by Customer Type")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 5.** Potential revenue uplift for each customer segment.")

    if {"Delivery_Speed_Importance","Quality_Certification_Priority"}.issubset(dff.columns):
        fig = px.scatter(dff, x="Delivery_Speed_Importance",
                         y="Quality_Certification_Priority",
                         color="Is_Converted" if "Is_Converted" in dff.columns else None,
                         title="Delivery vs Quality priority")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 6.** Relationship between delivery urgency and quality demands.")

    if "CRM_Engagement_Score" in dff.columns:
        fig = px.histogram(dff, x="CRM_Engagement_Score", nbins=10,
                           title="CRM Engagement Score Distribution",
                           color="Is_Converted" if "Is_Converted" in dff.columns else None,
                           barnorm="percent")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 7.** Leads with higher self‑rated engagement scores.")

    if {"Years_In_Operation","Monthly_Spend_Parts_INR"}.issubset(dff.columns):
        fig = px.scatter(dff, x="Years_In_Operation",
                         y="Monthly_Spend_Parts_INR",
                         color="Business_Type",
                         title="Years in Operation vs Monthly Spend")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 8.** Mature businesses’ spend patterns.")

    if "Average_Order_Frequency" in dff.columns:
        freq_count = dff["Average_Order_Frequency"].value_counts()
        fig = px.bar(freq_count, title="Order Frequency Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 9.** How often customers place orders.")

    if {"Delivery_Speed_Importance","Is_Converted"}.issubset(dff.columns):
        conv_rate = dff.groupby("Delivery_Speed_Importance")["Is_Converted"].mean()*100
        fig = px.line(conv_rate, markers=True,
                      title="Conversion vs Delivery Speed Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 10.** Time‑sensitive buyers convert more frequently.")

# =========================================================================
# 2. CLASSIFICATION
# =========================================================================
with tabs[1]:
    st.header("Classification Models")
    target = st.selectbox("Choose target label",
                          options=["Will_Trial_Bin","Is_Converted"],
                          help="Binary target for training.")
    if target not in df_processed.columns:
        st.error("Selected target not in data.")
        st.stop()

    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    # quick numeric/categorical split
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    test_size = st.slider("Test set %", 10, 40, 20, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y)

    results = {}
    for name, mdl in models.items():
        pipe = Pipeline([("prep", preproc),
                         ("clf", mdl)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = report["weighted avg"]
        results[name]["accuracy"] = report["accuracy"]

    res_df = (pd.DataFrame(results).T
              .loc[:, ["accuracy","precision","recall","f1-score"]]
              .round(3)
              .sort_values("f1-score", ascending=False))
    st.dataframe(res_df, use_container_width=True)

    # Confusion matrix toggle
    algo_choice = st.selectbox("Choose model for confusion matrix", list(models.keys()))
    if st.button("Show confusion matrix"):
        mdl = models[algo_choice]
        pipe = Pipeline([("prep", preproc),
                         ("clf", mdl)])
        pipe.fit(X_train, y_train)
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        fig_cm = px.imshow(cm, text_auto=True,
                           title=f"Confusion Matrix – {algo_choice}",
                           x=["Pred 0","Pred 1"], y=["True 0","True 1"])
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curves
    if st.checkbox("Show ROC‑AUC curves"):
        plt.figure()
        for name, mdl in models.items():
            pipe = Pipeline([("prep", preproc), ("clf", mdl)])
            pipe.fit(X_train, y_train)
            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
        plt.plot([0,1],[0,1],"--", color="k")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        st.pyplot(plt.gcf())

    # Upload new data for prediction
    st.subheader("Predict on new uploaded data")
    newfile = st.file_uploader("Upload CSV without target column",
                               type="csv", key="pred_upload")
    if newfile:
        new_df = pd.read_csv(newfile)
        best_model_name = res_df.index[0]
        st.info(f"Using best F1 model: **{best_model_name}**")
        pipe = Pipeline([("prep", preproc),
                         ("clf", models[best_model_name])])
        pipe.fit(X, y)  # train on full data
        preds = pipe.predict(new_df)
        out = new_df.copy()
        out["prediction"] = preds
        csv_bytes = out.to_csv(index=False).encode()
        b64 = base64.b64encode(csv_bytes).decode()
        st.download_button("Download predictions",
                           data=b64,
                           file_name="predictions.csv",
                           mime="text/csv")

# =========================================================================
# 3. CLUSTERING
# =========================================================================
with tabs[2]:
    st.header("K‑Means Customer Segmentation")
    k = st.slider("Number of clusters (k)", 2, 10, 4)

    X_clust = df_processed.select_dtypes(include="number")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    inertias = []
    k_range = range(2, 11)
    for k_i in k_range:
        km = KMeans(n_clusters=k_i, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    fig_elbow = px.line(x=list(k_range), y=inertias,
                        markers=True, title="Elbow Chart")
    fig_elbow.update_layout(xaxis_title="k", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)

    km_final = KMeans(n_clusters=k, n_init=25, random_state=42)
    df_processed["cluster"] = km_final.fit_predict(X_scaled)

    # Persona table
    persona = df_processed.groupby("cluster").agg({
        "Monthly_Spend_Parts_INR":"mean",
        "Expected_Monthly_Spend_NewVendor_INR":"mean",
        "Delivery_Speed_Importance":"mean",
        "Quality_Certification_Priority":"mean",
        "CRM_Engagement_Score":"mean"
    }).round(1).rename(columns={
        "Monthly_Spend_Parts_INR":"Mean Current Spend",
        "Expected_Monthly_Spend_NewVendor_INR":"Mean Expected Spend",
        "Delivery_Speed_Importance":"Delivery Priority",
        "Quality_Certification_Priority":"Quality Priority",
        "CRM_Engagement_Score":"CRM Score"
    })
    st.dataframe(persona, use_container_width=True)

    # Download cluster‑labelled data
    csv_cluster = df_processed.to_csv(index=False).encode()
    st.download_button("Download data with clusters",
                       data=csv_cluster,
                       file_name="clustered_data.csv",
                       mime="text/csv")

# =========================================================================
# 4. ASSOCIATION RULE MINING
# =========================================================================
with tabs[3]:
    st.header("Association Rules")
    col1 = st.selectbox("Column 1", cat_cols)
    col2 = st.selectbox("Column 2", [c for c in cat_cols if c!=col1])
    min_sup = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
    min_conf = st.number_input("Min confidence", 0.1, 1.0, 0.3, 0.05)

    # Prepare transaction matrix (one‑hot on two columns)
    basket = pd.get_dummies(df[[col1,col2]])
    frequent = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    top10 = (rules.sort_values("lift", ascending=False)
                   .head(10)[["antecedents","consequents","support","confidence","lift"]])
    st.dataframe(top10, use_container_width=True)

# =========================================================================
# 5. REGRESSION INSIGHTS
# =========================================================================
with tabs[4]:
    st.header("Quick Regression Insights")
    target_reg = st.selectbox("Choose numeric target",
                              options=["Expected_Monthly_Spend_NewVendor_INR",
                                       "Monthly_Spend_Parts_INR"])
    Xr = df_processed.drop(columns=[target_reg])
    yr = df_processed[target_reg]

    num_cols_r = Xr.select_dtypes(include="number").columns
    cat_cols_r = Xr.select_dtypes(exclude="number").columns

    prep_r = ColumnTransformer([
        ("num", StandardScaler(), num_cols_r),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_r)
    ])

    models_r = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeClassifier()  # treat as reg?
    }

    for name, mdl in models_r.items():
        pipe = Pipeline([("prep", prep_r), ("reg", mdl)])
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            Xr, yr, test_size=.2, random_state=42)
        pipe.fit(X_train_r, y_train_r)
        score = pipe.score(X_test_r, y_test_r)
        st.metric(label=f"{name} R²", value=f"{score:.3f}")

st.caption("© 2025 Auto‑Parts Analytics Dashboard")
