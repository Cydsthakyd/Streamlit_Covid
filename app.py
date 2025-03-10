import streamlit as st
import pandas as pd
import numpy as np
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Set page title and icon
st.set_page_config(page_title="Covid Analysis", page_icon="😷")

tab1, tab2, tab3 = st.tabs(["First Tab", "Second Tab", "Third Tab"])

with tab1:
# Load Data
    @st.cache_data
    def load_data():
        df = pd.read_csv("cleaned/covid_2.csv")  
        return df

# Train Model 
@st.cache_resource
def train_model(df):
    feature_columns = ["cases", "deaths", "hospitalized", "mutation_count", "transmission_rate", "mutation_transmission_ratio", "mutation_transmission_interaction", "case_fatality_rate", "variant_prevalence"]
    X = df[feature_columns]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["severity_level"])  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model, feature_columns


# Streamlit App Layout
st.title("COVID-19 Variant Severity Prediction & Data Analysis")
st.markdown("### Predict the severity level of a COVID-19 variant and explore dataset insights.")

df = load_data()
model, feature_columns = train_model(df)

# Sidebar Inputs
st.sidebar.header("Input Features")
input_features = {}
for feature in feature_columns:
    if df[feature].dtype in ['float64', 'int64']:
        input_features[feature] = st.sidebar.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

input_df = pd.DataFrame([input_features])

if st.sidebar.button("Predict Severity Level"):
    prediction = model.predict(input_df)
    severity_mapping = {1: "Mild", 2: "Moderate", 3: "Severe", 4: "Critical"}
    st.subheader(f"Predicted Severity Level: {severity_mapping.get(prediction[0], 'Unknown')}")

with tab2:
# Data Visualization
    st.markdown("### Exploratory Data Analysis")
    st.write("#### Distribution of Severity Levels")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=df['severity_level'], palette='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("### Raw Dataset")
    st.dataframe(df.head(21))

with tab3:
# Data Profiling
    st.markdown("### Automated Data Profiling Report")
    pr = df.profile_report()

    st.title("Profiling in Streamlit")
    st.write(df)
    st_profile_report(pr)

    
    
   

    
    