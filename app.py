# Analyzing Customer Behavior for E-commerce Insights

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from pathlib import Path

st.set_page_config(
    page_title="E-commerce Customer Insights Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("E-commerce Customer Behavior & Churn Analysis")
st.markdown("**Interactive Dashboard for Customer Insights and Predictive Analytics**")

# ==========================
# Upload CSV or Use Default
# ==========================

default_file = "e_commerce.csv"
df = None

if Path(default_file).exists():
    df = pd.read_csv(default_file)
else:
    uploaded_file = st.file_uploader(
        "Upload CSV (columns: customer_id, age, gender, location, frequency, avg_order_value, total_spent, avg_session_time_min, pages_viewed, product_category, recency_days, churned)",
        type=["csv"],
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Upload a CSV to begin or ensure it's named 'e_commerce.csv'")
        st.stop()

# ==========================
# Dataset Overview
# ==========================
with st.expander("📊 Dataset Overview"):
    st.write(df.head(7))
    st.write("Shape:", df.shape)
    st.write("Info:")
    buffer = df.info()
    st.text(buffer)
    st.write("Descriptive statistics:")
    st.write(df.describe())
    st.write("Duplicate rows:", df.duplicated().sum())

# ==========================
# Feature Engineering
# ==========================

# Create age_group if missing
if 'age_group' not in df.columns:
    df['age_group'] = pd.cut(df['age'], bins=[17, 25, 35, 50, 70], labels=['18-25', '26-35', '36-50', '51-70'])

df_1 = df.copy()
df_1['frequency_1'] = (df_1['frequency'] > 0).astype(int)  # Purchased at least once
df_1['high_value_customer'] = (df_1['total_spent'] > df_1['total_spent'].median()).astype(int)  # Above median

# Only include columns that exist for one-hot encoding
dummy_cols = [col for col in ['product_category','location','gender','age_group'] if col in df_1.columns]
df_1d = pd.get_dummies(df_1, columns=dummy_cols, drop_first=True)

# ==========================
# Exploratory Data Analysis
# ==========================
st.header("Exploratory Data Analysis")

tabs = st.tabs([
    "Product Category", "Gender & Product", "Age & Spending",
    "Churn Analysis", "Regional Insights", "Frequency & Spending"
])

# --- Tab 1: Product Category Pie ---
with tabs[0]:
    st.subheader("Overall Product Category Share")
    if 'product_category' in df.columns:
        cat_counts = df['product_category'].value_counts().reset_index()
        cat_counts.columns = ['product_category', 'count']
        fig = px.pie(cat_counts, values='count', names='product_category', title='Overall Product Category Share')
        fig.update_layout(width=900, height=500)
        st.plotly_chart(fig)
    else:
        st.info("Column 'product_category' not found in dataset.")

# --- Tab 2: Gender & Product ---
with tabs[1]:
    if {'product_category','gender'}.issubset(df.columns):
        st.subheader("Customer Gender Distribution by Product Category")
        fig = px.histogram(
            df,
            x="product_category",
            color="gender",
            barmode="group",
            title="Customer Gender Distribution by Product Category"
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

        st.subheader("Average Total Spent by Product Category and Gender")
        avg_spent = df.groupby(['product_category','gender'])['total_spent'].mean().reset_index()
        fig = px.bar(
            avg_spent,
            x='product_category',
            y='total_spent',
            color='gender',
            barmode='group',
            title='Average Total Spent by Product Category and Gender',
            labels={'total_spent':'Average Total Spent ($)','product_category':'Product Category'}
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

# --- Tab 3: Age & Spending ---
with tabs[2]:
    if {'age_group','gender','total_spent'}.issubset(df.columns):
        st.subheader("Age Distribution by Gender and Total Amount Spent")
        avg_age_spending = df.groupby(['age_group','gender'])['total_spent'].sum().reset_index()
        avg_age_spending = avg_age_spending.sort_values(by='total_spent', ascending=False)
        fig = px.bar(
            avg_age_spending,
            x='age_group',
            y='total_spent',
            color='gender',
            barmode='group',
            title='Age Distribution by Gender And Total Amount Spent',
            labels={'age':'Customer Age'}
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

# --- Tab 4: Churn Analysis ---
with tabs[3]:
    if {'product_category','churned'}.issubset(df.columns):
        st.subheader("Churn Rate by Product Category")
        churn_by_cat = df.groupby('product_category')['churned'].mean().sort_values(ascending=False)
        fig = px.bar(
            churn_by_cat.reset_index(),
            x='product_category',
            y='churned',
            title='Churn Rate by Product Category'
        )
        st.plotly_chart(fig)

# --- Tab 5: Regional Insights ---
with tabs[4]:
    if {'location','product_category'}.issubset(df.columns):
        st.subheader("Product Category Distribution Across Regions")
        region_cat = pd.crosstab(df['location'], df['product_category'])
        plt.figure(figsize=(10,6))
        sns.heatmap(region_cat, cmap='YlGnBu', annot=True, fmt='d')
        plt.title('Product Category Distribution Across Regions')
        st.pyplot(plt.gcf())

# --- Tab 6: Frequency & Spending ---
with tabs[5]:
    if {'gender','frequency','product_category','total_spent'}.issubset(df.columns):
        st.subheader("Shopping Frequency by Gender")
        fig = px.box(
            df,
            x='gender',
            y='frequency',
            color='gender',
            title='Shopping Frequency by Gender',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(width=800, height=500)
        st.plotly_chart(fig)

        st.subheader("Total Revenue by Product Category")
        total_spent_cat = df.groupby('product_category')['total_spent'].sum().reset_index().sort_values(by='total_spent', ascending=False)
        fig = px.bar(
            total_spent_cat,
            x='product_category',
            y='total_spent',
            color='product_category',
            title='Total Revenue by Product Category',
            labels={'total_spent':'Total Revenue ($)','product_category':'Product Category'},
            color_discrete_sequence=px.colors.qualitative.Dark2
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

# ==========================
# Predictive Modeling: Customer Churn
# ==========================
st.header("Predictive Modeling: Customer Churn")

# Only run if churn column exists
if 'churned' in df_1d.columns:
    X = df_1d.drop(columns=['customer_id','churned'], errors='ignore')
    y = df_1d['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    num_cols = ['frequency','avg_order_value','avg_session_time_min','pages_viewed','recency_days','total_spent']
    for col in num_cols:
        if col in X_train.columns:
            X_train[col] = scaler.fit_transform(X_train[[col]])
            X_test[col] = scaler.transform(X_test[[col]])

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    st.write({
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "ROC AUC Score": roc_auc
    })

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    st.write(f"Random Forest Cross-Validation ROC-AUC Scores: {cv_scores}")
    st.write(f"Mean ROC-AUC: {cv_scores.mean():.3f}")

# ==========================
# Business Insights
# ==========================
with st.expander("💡 Business Insights"):
    st.markdown("""
    - Customers with low frequency (≤1) and high recency_days (>90) are most likely to churn.
    - High-value customers (spend above median) have significantly lower churn — target them with loyalty rewards.
    - Product categories like 'Books' and 'Health' may have higher churn rates — consider reactivation promotions.
    - Females spend more in Fashion and Books; Males dominate Electronics and Home products.
    - Launch personalized campaigns: electronics bundles for male customers, grocery and fashion discounts for female customers.
    - Most customers are 51-70 years old; consider offers like free shipping for this group.
    - Focus logistics and advertising in Greater Accra and Ashanti for ROI.
    - Electronics generates highest revenue; Groceries and Fashion drive high volume.
    - Use Recency and Frequency to identify at-risk customers; targeted campaigns can reduce churn 5–10% and increase repeat purchases 15–20%.
    """)
