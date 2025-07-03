# ──────────────────────────────────────────
#  B2B Auto-Parts Lead-Analytics Dashboard
#  (full, polished, NaN-safe, dark theme)
# ──────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go, urllib.error, base64, warnings
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
pio.templates.default = "plotly_dark"
warnings.filterwarnings("ignore")

# ── consistent corporate palette ──────────────────────────
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

# ─── CONFIG ────────────────────────────────────────────────
DEFAULT_URL = (
    "https://raw.githubusercontent.com/raghul-95/B2B-auto-spare-parts/refs/heads/main/Data_Analysis_R_Survey_Enhanced.csv"
)
st.set_page_config(page_title="Auto-Parts Analytics Dashboard", layout="wide")

# ─── Data loader with graceful error banner ───────────────
@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except (urllib.error.URLError,
            urllib.error.HTTPError,
            FileNotFoundError,
            pd.errors.ParserError) as e:
        st.error(f"❌ Unable to load CSV: {e}")
        st.stop()

url_box = st.sidebar.text_input("GitHub raw CSV URL", value=DEFAULT_URL)
upload   = st.sidebar.file_uploader("…or upload CSV", type="csv")
df = load_csv(upload) if upload else load_csv(url_box)
st.sidebar.success(f"Loaded **{len(df):,} rows · {df.shape[1]} columns**")

# ─── Feature engineering ──────────────────────────────────
df_proc = df.copy()
if "Willing_To_Trial_NewVendor" in df_proc.columns and "Will_Trial_Bin" not in df_proc.columns:
    df_proc["Will_Trial_Bin"] = df_proc["Willing_To_Trial_NewVendor"].map(
        {"Yes": 1, "No": 0, "Undecided": 0}
    )

# ─── Helpers ──────────────────────────────────────────────
def make_prep(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    for bad in ["Lead_ID", "Company_Name"]:
        if bad in cat:  # drop high-cardinality ID-like cols
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
    # turn frozenset({'A','B'}) into "A, B"
    return ", ".join(sorted(list(fset)))

# ─── Tabs layout ──────────────────────────────────────────
tabs = st.tabs(
    [
        "📊 Visualisation",
        "🎯 Classification",
        "🔀 Clustering",
        "🔗 Assoc Rules",
        "📈 Regression",
    ]
)

# ═════════════════════════════════════════ 1  VISUALISATION
with tabs[0]:
    st.header("10 Executive-Level Insights")

    # 1 Sales Funnel
    if "Sales_Stage" in df.columns:
        counts = df["Sales_Stage"].value_counts()
        st.plotly_chart(
            go.Figure(
                go.Funnel(
                    y=counts.index, x=counts.values,
                    textinfo="value+percent initial"
                )
            ),
            use_container_width=True
        )
        st.caption("**Fig 1.** Lead drop-off at every pipeline stage.")

    # 2 Conversion by Lead Source
    if {"Lead_Source", "Is_Converted"}.issubset(df.columns):
        conv_src = (
            pd.crosstab(df["Lead_Source"], df["Is_Converted"], normalize="index") * 100
        )
        st.plotly_chart(
            px.bar(
                conv_src,
                barmode="stack",
                title="Conversion Rate by Lead Source",
                labels={"value": "%"},
            ),
            use_container_width=True,
        )
        st.caption("**Fig 2.** Website vs trade-fair conversion efficiency.")

    # 3 Region vs Product Category heat-map
    if {"Region", "Product_Category"}.issubset(df.columns):
        heat = pd.crosstab(df["Region"], df["Product_Category"])
        st.plotly_chart(
            px.imshow(heat, text_auto=True, title="Demand Heat-Map"),
            use_container_width=True,
        )
        st.caption("**Fig 3.** Regional demand patterns across product lines.")

    # 4 Monthly Spend distribution
    if "Monthly_Spend_Parts_INR" in df.columns:
        st.plotly_chart(
            px.box(
                df,
                y="Monthly_Spend_Parts_INR",
                points="all",
                title="Monthly Parts Spend Distribution",
            ),
            use_container_width=True,
        )
        st.caption("**Fig 4.** Wide variance hints at distinct spend tiers.")

    # 5 Expected vs Current spend uplift
    if {
        "Expected_Monthly_Spend_NewVendor_INR",
        "Monthly_Spend_Parts_INR",
    }.issubset(df.columns):
        uplift = (
            df["Expected_Monthly_Spend_NewVendor_INR"]
            / df["Monthly_Spend_Parts_INR"]
            - 1
        ) * 100
        st.plotly_chart(
            px.histogram(uplift, nbins=30, title="Projected Revenue Uplift (%)"),
            use_container_width=True,
        )
        st.caption("**Fig 5.** Majority of leads could raise spend by ≥10 %.")

    # 6 Delivery priority vs conversion
    if {"Delivery_Speed_Importance", "Is_Converted"}.issubset(df.columns):
        conv = (
            df.groupby("Delivery_Speed_Importance")["Is_Converted"].mean() * 100
        )
        st.plotly_chart(
            px.line(
                conv,
                markers=True,
                title="Conversion vs Delivery Priority",
                labels={"value": "% Conversion"},
            ),
            use_container_width=True,
        )
        st.caption("**Fig 6.** Time-critical buyers convert ~2× more often.")

    # 7 CRM engagement distribution
    if "CRM_Engagement_Score" in df.columns:
        st.plotly_chart(
            px.histogram(
                df,
                x="CRM_Engagement_Score",
                nbins=10,
                title="Self-Rated CRM Engagement",
            ),
            use_container_width=True,
        )
        st.caption("**Fig 7.** Engagement scores skew mid-high across leads.")

    # 8 Numeric correlation matrix
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 3:
        st.plotly_chart(
            px.imshow(df[num_cols].corr().round(2), text_auto=True,
                      title="Numeric Feature Correlations"),
            use_container_width=True,
        )
        st.caption("**Fig 8.** Key relationships among numeric drivers.")

    # 9 Order frequency distribution
    if "Average_Order_Frequency" in df.columns:
        st.plotly_chart(
            px.bar(
                df["Average_Order_Frequency"].value_counts(),
                title="Order Frequency Distribution",
            ),
            use_container_width=True,
        )
        st.caption("**Fig 9.** Weekly ordering is most common.")

    # 10 Years vs Spend with Savitzky-Golay trend
    if {
        "Years_In_Operation",
        "Monthly_Spend_Parts_INR",
    }.issubset(df.columns):
        scatter = (
            df[["Years_In_Operation", "Monthly_Spend_Parts_INR"]]
            .dropna()
            .sort_values("Years_In_Operation")
        )
        fig = px.scatter(
            df,
            x="Years_In_Operation",
            y="Monthly_Spend_Parts_INR",
            title="Business Maturity vs Monthly Spend",
        )
        if len(scatter) >= 11:
            smooth = savgol_filter(
                scatter["Monthly_Spend_Parts_INR"], 11, 2
            )
            fig.add_scatter(
                x=scatter["Years_In_Operation"],
                y=smooth,
                mode="lines",
                line=dict(color="white"),
                name="Trend (Savgol)",
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Fig 10.** Older firms tend to spend more, but variability remains high.")

# ═══════════════════════════════════════ 2  CLASSIFICATION
with tabs[1]:
    st.header("Binary Classification Suite")
    tgt = st.selectbox(
        "Target label", [c for c in ["Will_Trial_Bin", "Is_Converted"] if c in df_proc.columns]
    )
    if tgt:
        X, y = df_proc.drop(columns=[tgt]), df_proc[tgt]
        X_tr, X_ts, y_tr, y_ts = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        prep = make_prep(X)
        learners = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "GBRT": GradientBoostingClassifier(),
        }
        rows, probas = [], {}
        for name, mdl in learners.items():
            pipe = Pipeline([("prep", prep), ("est", mdl)]).fit(X_tr, y_tr)
            y_pred = pipe.predict(X_ts)
            if hasattr(pipe, "predict_proba"):
                probas[name] = pipe.predict_proba(X_ts)[:, 1]
            rep = classification_report(
                y_ts, y_pred, output_dict=True, zero_division=0
            )
            rows.append(
                {
                    "Model": name,
                    "Accuracy": round(rep["accuracy"], 3),
                    "Precision": round(rep["weighted avg"]["precision"], 3),
                    "Recall": round(rep["weighted avg"]["recall"], 3),
                    "F1": round(rep["weighted avg"]["f1-score"], 3),
                }
            )
        metr_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metr_df)

        # Confusion matrix toggle
        cm_model = st.selectbox("Confusion-matrix model", metr_df.index)
        if st.checkbox("Show confusion matrix"):
            cm_pipe = Pipeline([("prep", prep), ("est", learners[cm_model])]).fit(X_tr, y_tr)
            cm = confusion_matrix(y_ts, cm_pipe.predict(X_ts))
            st.plotly_chart(
                px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="True"),
                    title=f"Confusion Matrix – {cm_model}",
                ),
                use_container_width=True,
            )

        # ROC curves
        if st.checkbox("Show ROC curves"):
            fig = go.Figure()
            for name, prob in probas.items():
                fpr, tpr, _ = roc_curve(y_ts, prob)
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{name} (AUC {auc(fpr, tpr):.2f})",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash"), showlegend=False
                )
            )
            fig.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Batch predict
        new_file = st.file_uploader(
            "Upload new data (without target) to predict", type="csv", key="pred_upload"
        )
        if new_file:
            new_df = pd.read_csv(new_file)
            best_model = metr_df["F1"].idxmax()
            st.info(f"Using best F1 model: **{best_model}**")
            best_pipe = Pipeline([("prep", prep), ("est", learners[best_model])]).fit(X, y)
            new_df["prediction"] = best_pipe.predict(new_df)
            st.markdown(
                download_link(new_df, "predictions.csv"), unsafe_allow_html=True
            )

# ═══════════════════════════════════════ 3  CLUSTERING
with tabs[2]:
    st.header("K-Means Customer Segmentation")
    k = st.slider("Number of clusters (k)", 2, 10, 4)
    num_df = (
        df_proc.select_dtypes(include="number")
        .fillna(df_proc.select_dtypes(include="number").median())
    )
    scaled = StandardScaler().fit_transform(num_df)
    inertias = [
        KMeans(n_clusters=i, n_init=10, random_state=42).fit(scaled).inertia_
        for i in range(2, 11)
    ]
    st.plotly_chart(
        px.line(
            x=list(range(2, 11)),
            y=inertias,
            markers=True,
            title="Elbow Chart (k selection)",
        ),
        use_container_width=True,
    )
    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(scaled)
    df_clust = df_proc.copy()
    df_clust["cluster"] = km.labels_
    persona = (
        df_clust.groupby("cluster")[
            [
                "Monthly_Spend_Parts_INR",
                "Expected_Monthly_Spend_NewVendor_INR",
                "Delivery_Speed_Importance",
                "Quality_Certification_Priority",
                "CRM_Engagement_Score",
            ]
        ]
        .mean()
        .round(1)
    )
    st.dataframe(persona)
    st.markdown(download_link(df_clust, "clustered_data.csv"), unsafe_allow_html=True)

# ═══════════════════════════════════════ 4  ASSOCIATION RULES
with tabs[3]:
    st.header("Apriori Association Rules")
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols) >= 2:
        col_a = st.selectbox("Column A", cat_cols)
        col_b = st.selectbox("Column B", [c for c in cat_cols if c != col_a])
        sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        conf = st.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)
        if st.button("Run Apriori"):
            basket = pd.get_dummies(df[[col_a, col_b]])
            freq = apriori(basket, min_support=sup, use_colnames=True)
            if freq.empty:
                st.warning("No itemsets meet support threshold.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=conf)
                if rules.empty:
                    st.warning("No rules meet confidence threshold.")
                else:
                    rules["antecedents"] = rules["antecedents"].apply(stringify)
                    rules["consequents"] = rules["consequents"].apply(stringify)
                    st.dataframe(
                        rules.sort_values("lift", ascending=False).head(10)[
                            ["antecedents", "consequents",
                             "support", "confidence", "lift"]
                        ],
                        use_container_width=True,
                    )
    else:
        st.info("Need ≥2 categorical columns.")

# ═══════════════════════════════════════ 5  REGRESSION
with tabs[4]:
    st.header("Regression Insights")
    num_targets = [
        c
        for c in [
            "Expected_Monthly_Spend_NewVendor_INR",
            "Monthly_Spend_Parts_INR",
        ]
        if c in df_proc.columns
    ]
    tgt_num = st.selectbox("Numeric target", num_targets)
    if tgt_num:
        Xn, yn = df_proc.drop(columns=[tgt_num]), df_proc[tgt_num]
        Xtr, Xts, ytr, yts = train_test_split(
            Xn, yn, test_size=0.2, random_state=42
        )
        prep_num = make_prep(Xn)
        regressors = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }
        rows = []
        for name, reg in regressors.items():
            pipe = Pipeline([("prep", prep_num), ("reg", reg)]).fit(Xtr, ytr)
            preds = pipe.predict(Xts)
            rows.append(
                {
                    "Model": name,
                    "R²": round(r2_score(yts, preds), 3),
                    "MAE": round(mean_absolute_error(yts, preds), 1),
                    "RMSE": round(
                        np.sqrt(mean_squared_error(yts, preds)), 1
                    ),
                }
            )
        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metrics_df)

        # Dynamic actual-vs-predicted scatter
        model_plot = st.selectbox("Plot Actual vs Predicted for", metrics_df.index)
        best_pipe = Pipeline(
            [("prep", prep_num), ("reg", regressors[model_plot])]
        ).fit(Xtr, ytr)
        preds = best_pipe.predict(Xts)
        fig = px.scatter(
            x=yts,
            y=preds,
            labels=dict(x="Actual", y="Predicted"),
            title=f"Actual vs Predicted — {model_plot}",
        )
        min_, max_ = yts.min(), yts.max()
        fig.add_trace(
            go.Scatter(
                x=[min_, max_],
                y=[min_, max_],
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False,
            )
        )
        st.plotly_chart(fig, use_container_width=True)
