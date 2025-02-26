## Data Collection & Preprocessing
- The key features used in the analysis are:
Cases, Deaths, Hospitalized, Mutation Count, Transmission Rate, Mutation-Transmission Ratios, Case Fatality Rate, Variant Prevalence.
- The target variable for prediction is severity_level, classified into Mild, Moderate, Severe, and Critical categories.

## Machine Learning Model: Random Forest Classifier
- The dataset is split into training (80%) and testing (20%) sets using train_test_split from sklearn.
-  A Random Forest Classifier is trained on the feature set to predict the severity level.

## Streamlit Web Application

### 📌 Tab 1: COVID-19 Severity Prediction
- Users can input real-time values for features such as cases, deaths, hospitalization rate, etc. using slider widgets in the sidebar.
- After inputting data, the trained model predicts the severity level using a classification mapping (Mild, Moderate, Severe, Critical).

### 📌 Tab 2: Exploratory Data Analysis (EDA)
- Distribution of Severity Levels: A Seaborn count plot visualizes the frequency of severity levels.
- Correlation Heatmap: A heatmap is generated to analyze the relationship between features, helping identify the strongest predictors of severity.
- Raw Dataset View: Displays the first 21 rows of the dataset for reference.

### 📌 Tab 3: Automated Data Profiling
- The profiling tool computes summary statistics, missing values, feature correlations, and distributions to help understand dataset quality.
