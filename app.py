
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from scipy.signal import savgol_filter
from mlxtend.frequent_patterns import apriori, association_rules
import base64, urllib.error, warnings
warnings.filterwarnings('ignore')

DEFAULT_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/Data_Analysis_R_Survey_Enhanced.csv"
st.set_page_config(page_title="Auto‑Parts Dashboard v7", layout="wide")

@st.cache_data(ttl=3600)
def load_csv(src):
    try:
        return pd.read_csv(src)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        st.error(f"❌ Could not load CSV: {e}")
        st.stop()

url = st.sidebar.text_input("Raw CSV URL", value=DEFAULT_URL)
upload = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_csv(upload) if upload else load_csv(url)
st.sidebar.success(f"{len(df):,} rows")

df_proc = df.copy()
if 'Willing_To_Trial_NewVendor' in df_proc.columns and 'Will_Trial_Bin' not in df_proc.columns:
    df_proc['Will_Trial_Bin'] = df_proc['Willing_To_Trial_NewVendor'].map({'Yes':1,'No':0,'Undecided':0})

def pre(X):
    num = X.select_dtypes(include='number').columns
    cat = X.select_dtypes(exclude='number').columns
    for bad in ['Lead_ID','Company_Name']:
        if bad in cat: cat = cat.drop(bad)
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),
                              ('cat',OneHotEncoder(handle_unknown='ignore'),cat)])

def dlink(df,name):
    csv=df.to_csv(index=False).encode(); b=base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b}" download="{name}">Download CSV</a>'

tabs = st.tabs(['EDA','Classification'])

# ----- EDA -----
with tabs[0]:
    st.header("Exploratory Insights")
    # Funnel
    if 'Sales_Stage' in df.columns:
        counts=df['Sales_Stage'].value_counts()
        fig=go.Figure(go.Funnel(y=counts.index, x=counts.values, textinfo='value+percent initial'))
        st.plotly_chart(fig, use_container_width=True)

    # Years vs spend with Savitzky‑Golay smooth line
    if {'Years_In_Operation','Monthly_Spend_Parts_INR'}.issubset(df.columns):
        fig = px.scatter(df, x='Years_In_Operation', y='Monthly_Spend_Parts_INR',
                         color='Business_Type' if 'Business_Type' in df.columns else None,
                         title='Years in Operation vs Monthly Spend')
        tmp=df[['Years_In_Operation','Monthly_Spend_Parts_INR']].dropna().sort_values('Years_In_Operation')
        if len(tmp)>=11:
            y_smooth=savgol_filter(tmp['Monthly_Spend_Parts_INR'], 11, 2)
            fig.add_scatter(x=tmp['Years_In_Operation'], y=y_smooth,
                            mode='lines', name='Savgol Trend',
                            line=dict(color='white', width=2))
        st.plotly_chart(fig, use_container_width=True)

# ----- Classification minimal (for brevity) -----
with tabs[1]:
    st.header("Classification (RF demo)")
    if 'Is_Converted' in df_proc.columns:
        X,y = df_proc.drop(columns=['Is_Converted']), df_proc['Is_Converted']
        Xtr,Xts,ytr,yts = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        pipe=Pipeline([('pre',pre(X)),('rf',RandomForestClassifier(random_state=42))]).fit(Xtr,ytr)
        ypred=pipe.predict(Xts)
        rep=classification_report(yts,ypred,output_dict=True)
        st.write(pd.DataFrame(rep).T[['precision','recall','f1-score']])
