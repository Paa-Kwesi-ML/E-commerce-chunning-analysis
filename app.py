# E-commerce Customer Behavior & Churn Analysis Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# --------------------------
# Page Config
st.set_page_config(
    page_title="E-commerce Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Title
st.title("E-commerce Customer Behavior & Churn Analysis")

# --------------------------
# Upload CSV Option
st.sidebar.header("Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Choose your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
else:
    st.warning("No file uploaded. Using sample generated dataset...")
    np.random.seed(42)
    n_customers = 5000
    customer_id = np.arange(1, n_customers + 1)
    age = np.random.randint(18, 70, size=n_customers)
    gender = np.random.choice(['Male', 'Female'], size=n_customers, p=[0.5, 0.5])
    location = np.random.choice(['Greater Accra', 'Ashanti Region', 'Eastern Region', 
                                 'Northern Region', 'Savana Region', 'Western Region', 
                                 'Volta Region'], size=n_customers, 
                                p=[0.35,0.2,0.13,0.11,0.10,0.06,0.05])
    frequency = np.random.poisson(lam=2.0, size=n_customers)
    avg_order_value = np.random.normal(loc=40, scale=25, size=n_customers).clip(5,500)
    total_spent = (frequency * avg_order_value).round(2)
    avg_session_time = np.random.exponential(scale=6, size=n_customers)
    pages_viewed = np.random.poisson(lam=5, size=n_customers).clip(1)
    product_category = np.random.choice(['Electronics', 'Fashion', 'Furniture', 'Groceries', 
                                         'Health', 'Books'], size=n_customers, 
                                        p=[0.25,0.2,0.15,0.2,0.1,0.1])
    recency = []
    for f in frequency:
        if f == 0:
            recency.append(np.random.randint(120, 720))
        else:
            recency.append(int(np.random.exponential(scale=60)))
    churned = [1 if (r > 90 and f <= 1) else 0 for r,f in zip(recency, frequency)]

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "location": location,
        "frequency": frequency,
        "avg_order_value": avg_order_value.round(2),
        "total_spent": total_spent,
        "avg_session_time_min": avg_session_time.round(2),
        "pages_viewed": pages_viewed,
        "product_category": product_category,
        "recency_days": recency,
        "churned": churned
    })

# --------------------------
# Sidebar filters
st.sidebar.header("Filter Data")
selected_locations = st.sidebar.multiselect("Select Locations", options=df['location'].unique(), default=df['location'].unique())
selected_categories = st.sidebar.multiselect("Select Product Categories", options=df['product_category'].unique(), default=df['product_category'].unique())
filtered_df = df[df['location'].isin(selected_locations) & df['product_category'].isin(selected_categories)]

st.subheader("Sample Data")
st.dataframe(filtered_df.head(10))

# --------------------------
# Product Category Pie Chart
st.markdown("### Overall Product Category Share")
cat_counts = filtered_df['product_category'].value_counts().reset_index()
cat_counts.columns = ['product_category', 'count']
fig = px.pie(
    cat_counts,
    names='product_category',
    values='count',
    title='Overall Product Category Share'
)
fig.update_layout(width=800, height=500)
st.plotly_chart(fig)

# --------------------------
# Product Category by Gender
st.markdown("### Customer Gender Distribution by Product Category")
fig = px.histogram(
    filtered_df,
    x="product_category",
    color="gender",
    barmode="group",
    title="Customer Gender Distribution by Product Category"
)
fig.update_layout(width=800, height=600, xaxis_title='Product Category', yaxis_title='Number of Customers')
st.plotly_chart(fig)

# --------------------------
# Average Total Spent by Product Category and Gender
st.markdown("### Average Total Spent by Product Category and Gender")
avg_spent = filtered_df.groupby(['product_category','gender'])['total_spent'].mean().reset_index()
fig = px.bar(
    avg_spent,
    x='product_category',
    y='total_spent',
    color='gender',
    barmode='group',
    title='Average Total Spent by Product Category and Gender',
    labels={'total_spent':'Average Total Spent ($)','product_category':'Product Category'}
)
fig.update_layout(width=800, height=600, title_x=0.5)
st.plotly_chart(fig)

# --------------------------
# Age Distribution by Gender and Total Amount Spent
st.markdown("### Age Distribution by Gender and Total Amount Spent")
df['age_group']= pd.cut(df['age'], bins=[17,25,35,50,70], labels=['18-25','26-35','36-50','51-70'])
avg_age_spending_ = df.groupby(['age_group','gender'])['total_spent'].sum().reset_index()
age_gd = avg_age_spending_.sort_values(by='total_spent', ascending=False)
fig = px.bar(
    age_gd,
    x='age_group',
    y='total_spent',
    color='gender',
    barmode='group',
    title='Age Distribution by Gender and Total Amount Spent'
)
fig.update_layout(width=800, height=600, title_x=0.5)
st.plotly_chart(fig)

# --------------------------
# Churn Rate by Product Category
st.markdown("### Churn Rate by Product Category")
churn_by_cat = filtered_df.groupby('product_category')['churned'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=churn_by_cat.index, y=churn_by_cat.values, palette='viridis', ax=ax)
ax.set_title('Churn Rate by Product Category', fontsize=14, weight='bold')
ax.set_ylabel('Churn Rate')
ax.set_xlabel('Product Category')
st.pyplot(fig)

# --------------------------
# Heatmap: Product Category Distribution Across Regions
st.markdown("### Product Category Distribution Across Regions")
region_cat = pd.crosstab(filtered_df['location'], filtered_df['product_category'])
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(region_cat, cmap='YlGnBu', annot=True, fmt='d', ax=ax)
ax.set_title('Product Category Distribution Across Regions', fontsize=14, weight='bold')
ax.set_xlabel('Product Category')
ax.set_ylabel('Region')
st.pyplot(fig)

# --------------------------
# Shopping Frequency by Gender
st.markdown("### Shopping Frequency by Gender")
fig = px.box(
    filtered_df,
    x='gender',
    y='frequency',
    color='gender',
    title='Shopping Frequency by Gender',
    labels={'frequency':'Number of Purchases'},
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig.update_layout(width=800, height=500, title_x=0.5)
st.plotly_chart(fig)

# --------------------------
# Total Revenue by Product Category
st.markdown("### Total Revenue by Product Category")
total_spent_cat = filtered_df.groupby('product_category')['total_spent'].sum().reset_index().sort_values(by='total_spent', ascending=False)
fig = px.bar(
    total_spent_cat,
    x='product_category',
    y='total_spent',
    color='product_category',
    title='Total Revenue by Product Category',
    labels={'total_spent':'Total Revenue ($)','product_category':'Product Category'},
    color_discrete_sequence=px.colors.qualitative.Dark2
)
fig.update_layout(width=800, height=600, title_x=0.5)
st.plotly_chart(fig)

# --------------------------
# Feature Engineering for Churn Model
df_1 = filtered_df.copy()
df_1['frequency_1'] = (df_1['frequency'] > 0).astype(int)
df_1['high_value_customer'] = (df_1['total_spent'] > df_1['total_spent'].median()).astype(int)

df_1d = pd.get_dummies(df_1, columns=['product_category','location','gender','age_group'], drop_first=True)

# --------------------------
# Modeling: Predicting Customer Churn
X = df_1d.drop(columns=['customer_id','churned'])
y = df_1d['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_cols = ['frequency','avg_order_value','avg_session_time_min','pages_viewed','recency_days','total_spent']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

st.markdown("### Model Evaluation Metrics")
st.write({
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "ROC AUC Score": roc_auc
})

# --------------------------
# Cross-validation ROC-AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
st.write("### Random Forest Cross-Validation ROC-AUC Scores")
st.write(cv_scores)
st.write(f"Mean ROC-AUC: {cv_scores.mean():.3f}")
