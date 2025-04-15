# ------------------------------------------------------------------------------
# MIDTERM PROJECT: DATA SCIENCE WITH PYTHON
# ------------------------------------------------------------------------------
#
# 1. PROJECT OVERVIEW AND OBJECTIVES
#
#   1.1 Select a Dataset
#       - Requirement: Choose a dataset not used in class. 
#         It can come from public data repositories (e.g., Kaggle, 
#         UCI Machine Learning Repository, government portals).
#       - Tip: Pick a dataset that interests your group and has 
#         enough complexity to warrant cleaning, EDA, and modeling.
#
#   1.2 Primary Goal
#       - Apply data science concepts—data cleaning, visualization, modeling,
#         and evaluation—to gain insights and showcase Python proficiency.
#
# ------------------------------------------------------------------------------
# 2. PROJECT TASKS IN DETAIL
# ------------------------------------------------------------------------------
#
# 2.1 Acquire, Clean, and Preprocess Data
#
#   (a) Data Acquisition
#       - Identify your data source: file-based (CSV, JSON), database, API, etc.
#       - Document how you obtained it. For example, if from an API, show the request.
#
#   (b) Data Cleaning
#       - Tasks: Handle missing values, remove duplicates, correct invalid entries.
#       - Python Tools: pandas methods (isnull, dropna, fillna, etc.).
#       - Tips: Always justify your decisions, e.g., why dropping vs. imputing missing values.
#
#   (c) Data Preprocessing
#       - Requirement: Use at least 2 preprocessing techniques 
#         (scaling, encoding, feature engineering, etc.).
#       - Tips: Ensure numeric vs. categorical variables are appropriately transformed.
#
# ------------------------------------------------------------------------------
# 2.2 Perform Exploratory Data Analysis (EDA) and Visualize Key Insights
#
#   (a) Exploratory Data Analysis
#       - Compute basic stats (mean, median, std, etc.).
#       - Identify correlations, outliers, or data imbalances.
#       - Use pandas describe(), info(), corr() for an overview.
#
#   (b) Data Visualization
#       - Requirement: At least 3 different visualization techniques (histogram, 
#         scatter plot, box plot, heatmap, etc.).
#       - Tips: Use clear labels, titles, and legends. Let visuals drive your EDA narrative.
#
# ------------------------------------------------------------------------------
# 2.3 Build and Evaluate a Machine Learning Model
#
#   (a) Model Building
#       - Requirement: At least 2 different ML algorithms 
#         (e.g., Logistic Regression, Random Forest, Linear Regression, etc.).
#       - Tips: Match the algorithm type to your target variable 
#         (classification vs. regression).
#
#   (b) Model Evaluation
#       - Requirement: At least 2 different evaluation metrics 
#         (accuracy, precision/recall, F1, RMSE, MAE, etc.).
#       - Tips: Present numeric results and interpret them in plain English. 
#         Consider basic hyperparameter tuning.
#
# ------------------------------------------------------------------------------
# 3. DELIVERABLES
# ------------------------------------------------------------------------------
#
#   3.1 Code
#       - A well-commented Python script or Jupyter Notebook with:
#         * Data acquisition, cleaning, preprocessing
#         * EDA and visualizations
#         * Model building, training, and evaluation
#       - Ensure reproducibility. Include data or instructions to access it.
#
#   3.2 Report (Due in 3 Weeks)
#       - Structure:
#         1) Introduction to the Dataset
#         2) Data Cleaning & Preprocessing Steps
#         3) EDA & Key Insights
#         4) Model Building & Evaluation
#         5) Conclusion
#         6) References (if any)
#
# ------------------------------------------------------------------------------
# 4. TEAM COLLABORATION AND SUBMISSION TIPS
# ------------------------------------------------------------------------------
#
#   (a) Group Roles
#       - Decide early who focuses on which aspect: data cleaning, modeling, etc.
#       - Use Git or a similar VCS to merge changes and maintain a single codebase.
#
#   (b) Progress Milestones
#       - Week 1: Finalize dataset, do initial cleaning and EDA.
#       - Week 2: Refine preprocessing, build and evaluate at least one model.
#       - Week 3: Complete second model, finalize visualizations, write report.
#
#   (c) Version Control
#       - Commit frequently, use branches for different tasks, review each other's code.
#
#   (d) Polish and Professionalism
#       - Keep code readable and well-structured (clear variable names, function docstrings).
#       - Proofread your report, ensure visualizations are well-labeled.
#
# ------------------------------------------------------------------------------
# 5. PUTTING IT ALL TOGETHER
# ------------------------------------------------------------------------------
#
# By following this guide, your group will:
#   - Acquire data from a new source and thoroughly clean it.
#   - Preprocess it (e.g., scaling, encoding, feature engineering) as needed.
#   - Conduct an informative EDA with multiple visualizations.
#   - Train at least two machine learning models, evaluate them with multiple metrics.
#   - Compile findings in a concise, well-organized final report.
#
# Good luck with your data exploration and modeling!
# ------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spotify Streaming Data Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the 'Spotify_2024_Global_Streaming_Data.csv' file
df = pd.read_csv(r"C:\Users\sweth\OneDrive\Documents\GitHub\MSITM.6341\Group.Projects\Spotify_2024_Global_Streaming_Data.csv")

# 2. Clean column names: lowercase, strip whitespace, replace spaces with underscores, remove parentheses
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df.columns = df.columns.str.replace('(', '', regex=False).str.replace(')', '', regex=False)
print("Initial shape:", df.shape)
print(df.head(), "\n")

# 3. Remove duplicates and drop rows with missing values in specific columns
df.drop_duplicates(inplace=True)
print("Missing values after removing duplicates:\n", df.isnull().sum(), "\n")
df.dropna(subset=['streams_last_30_days_millions', 'avg_stream_duration_min'], inplace=True)

# 4. Clean numeric columns with commas and convert to float (if any)
if df['streams_last_30_days_millions'].dtype == 'object':
    df['streams_last_30_days_millions'] = df['streams_last_30_days_millions'].str.replace(',', '').astype(float)

# 5. If 'release_date' does not exist, set 'release_year' to 2024
if 'release_date' in df.columns:
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
else:
    df['release_year'] = 2024

# 6. Apply MinMaxScaler to 'avg_stream_duration_min' to create 'track_length_scaled'
scaler = MinMaxScaler()
df['track_length_scaled'] = scaler.fit_transform(df[['avg_stream_duration_min']])

# 7. Perform Exploratory Data Analysis (EDA)
print("Descriptive Statistics:\n", df.describe())
print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))

# (a) Histogram of streams_last_30_days_millions
plt.figure(figsize=(8, 5))
sns.histplot(df['streams_last_30_days_millions'], bins=30, kde=True)
plt.title("Distribution of Streams (Last 30 Days)")
plt.xlabel("Streams (Millions)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# (b) Boxplot of avg_stream_duration_min by genre
if 'genre' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='genre', y='avg_stream_duration_min')
    plt.xticks(rotation=45)
    plt.title("Average Stream Duration by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Avg Stream Duration (min)")
    plt.tight_layout()
    plt.show()

# (c) Correlation heatmap for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# (d) Top 10 streamed albums (by streams in last 30 days)
if 'album' in df.columns:
    top_songs = df.sort_values(by='streams_last_30_days_millions', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='streams_last_30_days_millions', y='album', data=top_songs, palette='mako')
    plt.title("Top 10 Streamed Albums (Last 30 Days)")
    plt.xlabel("Streams (Millions)")
    plt.ylabel("Album")
    plt.tight_layout()
    plt.show()

# (e) Top 10 most frequent artists (by number of appearances)
if 'artist' in df.columns:
    top_artists = df['artist'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_artists.values, y=top_artists.index, palette='rocket')
    plt.title("Top 10 Most Featured Artists")
    plt.xlabel("Number of Appearances")
    plt.ylabel("Artist")
    plt.tight_layout()
    plt.show()

# 8. Select features and target for modeling
features = ['track_length_scaled', 'release_year']
X = df[features]
y = df['streams_last_30_days_millions']

# 9. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Train models: Linear Regression and Random Forest Regressor
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# 11. Evaluate models
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

evaluate_model("Linear Regression", y_test, pred_lr)
evaluate_model("Random Forest", y_test, pred_rf)

# 12. Output Linear Regression coefficients
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lr.coef_})
print("\nLinear Regression Coefficients:")
print(coef_df)
