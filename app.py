
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEFAULT_URL = 'https://raw.githubusercontent.com/<user>/<repo>/main/Data_Analysis_R_Survey_Enhanced.csv'
st.set_page_config(page_title='Auto‑Parts Dashboard', layout='wide')


@st.cache_data(ttl=3600)
def load_csv(src):
    return pd.read_csv(src)


# -------------------------------------------------
# SIDEBAR – DATA SOURCE
# -------------------------------------------------
url = st.sidebar.text_input('GitHub raw CSV URL', DEFAULT_URL)
uploaded = st.sidebar.file_uploader('Upload CSV', type='csv')

try:
    df = load_csv(uploaded) if uploaded else load_csv(url)
except Exception as e:
    st.error(f'❌ Unable to read CSV: {e}')
    st.stop()

st.sidebar.success(f'Loaded {len(df):,} rows • {df.shape[1]} cols')

# -------------------------------------------------
# BASIC FEATURE ENGINEERING
# -------------------------------------------------
df_proc = df.copy()
if 'Willing_To_Trial_NewVendor' in df_proc.columns and 'Will_Trial_Bin' not in df_proc.columns:
    df_proc['Will_Trial_Bin'] = df_proc['Willing_To_Trial_NewVendor'].map({'Yes': 1, 'No': 0, 'Undecided': 0})


def make_preprocessor(X):
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    for noisy in ['Lead_ID', 'Company_Name']:
        if noisy in cat_cols:
            cat_cols.remove(noisy)

    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])


# -------------------------------------------------
# TABS
# -------------------------------------------------
tabs = st.tabs(['📊 EDA', '🎯 Classification', '🔀 Clustering', '🔗 Assoc Rules', '📈 Regression'])

# =================================================
# 1. EDA
# =================================================
with tabs[0]:
    st.header('Exploratory Data Analysis')
    if 'Is_Converted' in df.columns:
        conv = df['Is_Converted'].value_counts(normalize=True) * 100
        st.plotly_chart(px.pie(conv, values=conv.values, names=conv.index,
                                title='Conversion Split (%)'), use_container_width=True)

# =================================================
# 2. Classification
# =================================================
with tabs[1]:
    st.header('Binary Classification')
    tgt_options = [c for c in ['Will_Trial_Bin', 'Is_Converted'] if c in df_proc.columns]
    if not tgt_options:
        st.warning('No binary target columns available.')
    else:
        target = st.selectbox('Target label', tgt_options)
        if st.button('Train models'):
            X = df_proc.drop(columns=[target])
            y = df_proc[target]
            preproc = make_preprocessor(X)
            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)

            models = {
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'GBRT': GradientBoostingClassifier(random_state=42)
            }

            rows = []
            for name, mdl in models.items():
                pipe = Pipeline([('prep', preproc), ('model', mdl)])
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_ts)
                report = classification_report(y_ts, y_pred, output_dict=True, zero_division=0)
                acc_val = report.get('accuracy', accuracy_score(y_ts, y_pred))
                rows.append({
                    'Model': name,
                    'Accuracy': round(acc_val, 3),
                    'Precision': round(report['weighted avg']['precision'], 3),
                    'Recall': round(report['weighted avg']['recall'], 3),
                    'F1': round(report['weighted avg']['f1-score'], 3)
                })

            st.dataframe(pd.DataFrame(rows).set_index('Model'))

# =================================================
# 3. Clustering
# =================================================
with tabs[2]:
    st.header('K‑Means Clustering')
    if st.checkbox('Run clustering'):
        k = st.slider('k', 2, 10, 4)
        num_cols = df_proc.select_dtypes(include='number').columns
        scaler = Pipeline([('imp', SimpleImputer(strategy='median')),
                           ('sc', StandardScaler())])
        X_scaled = scaler.fit_transform(df_proc[num_cols])
        km = KMeans(n_clusters=k, random_state=42, n_init=25).fit(X_scaled)
        df_proc['cluster'] = km.labels_
        st.dataframe(df_proc.groupby('cluster')[num_cols].mean().round(1))

# =================================================
# 4. Association Rules
# =================================================
with tabs[3]:
    st.header('Association Rule Mining')
    cats = df.select_dtypes(exclude='number').columns.tolist()
    if len(cats) < 2:
        st.info('Need at least two categorical columns.')
    else:
        col1 = st.selectbox('Column 1', cats, key='ar1')
        col2 = st.selectbox('Column 2', [c for c in cats if c != col1], key='ar2')
        min_sup = st.slider('Min support', 0.01, 0.2, 0.05, 0.01)
        min_conf = st.slider('Min confidence', 0.1, 0.9, 0.3, 0.05)

        if st.button('Generate rules'):
            basket = pd.get_dummies(df[[col1, col2]])
            freq = apriori(basket, min_support=min_sup, use_colnames=True)
            if freq.empty:
                st.warning('No itemsets at this support. Lower threshold or change columns.')
            else:
                rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
                if rules.empty:
                    st.warning('No association rules meet confidence threshold.')
                else:
                    st.dataframe(rules.sort_values('lift', ascending=False)
                                 .head(10)[['antecedents', 'consequents',
                                            'support', 'confidence', 'lift']],
                                 use_container_width=True)

# =================================================
# 5. Regression
# =================================================
with tabs[4]:
    st.header('Ridge Regression')
    targets = [c for c in ['Expected_Monthly_Spend_NewVendor_INR',
                           'Monthly_Spend_Parts_INR'] if c in df_proc.columns]
    if not targets:
        st.info('No numeric targets available.')
    else:
        target_r = st.selectbox('Numeric target', targets)
        if st.button('Train Ridge'):
            Xr = df_proc.drop(columns=[target_r]); yr = df_proc[target_r]
            pre_r = make_preprocessor(Xr)
            X_tr, X_ts, y_tr, y_ts = train_test_split(Xr, yr, test_size=0.2, random_state=42)
            pipe = Pipeline([('prep', pre_r), ('ridge', Ridge(alpha=1.0))])
            pipe.fit(X_tr, y_tr)
            st.metric('R²', f'{pipe.score(X_ts, y_ts):.3f}')

st.caption('Stable app – NaN safe & KeyError guarded')
