import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

st.set_page_config(page_title="Cyberattack Analysis", layout="wide")

# Title
st.title("🔐 Cyberattack Threat Analysis (2015–2024)")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')
    return df

df = load_data()

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["📊 Dataset Overview", "📈 Visualizations", "📌 K-Means Clustering", "🧠 Naive Bayes Classification"])

# 1. Dataset Overview
if menu == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Descriptive Statistics")
    st.write(df.drop(columns=['Year']).describe())

    st.subheader("Value Counts (Categorical Columns)")
    for col in df.select_dtypes('object').columns:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())

# 2. Visualizations
elif menu == "📈 Visualizations":
    st.header("📈 Visual Explorations")

    # Attack Type and Target Industry Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution: Attack Type")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, y='Attack Type', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Distribution: Target Industry")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, y='Target Industry', ax=ax2)
        st.pyplot(fig2)

    # Numerical Histograms
    st.subheader("Distributions of Numerical Features")
    numeric_cols = [
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

# 3. K-Means Clustering
elif menu == "📌 K-Means Clustering":
    st.header("📌 K-Means Clustering")

    # Prepare for clustering
    df_kmeans = df.copy()
    df_kmeans = pd.get_dummies(df_kmeans, drop_first=True)

    numeric_cols = [
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]
    scaler = StandardScaler()
    df_kmeans[numeric_cols] = scaler.fit_transform(df_kmeans[numeric_cols])

    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(df_kmeans)
        sil_scores.append(silhouette_score(df_kmeans, clusters))

    optimal_k = range(2, 9)[sil_scores.index(max(sil_scores))]
    st.write(f"✅ **Optimal Number of Clusters:** {optimal_k}")

    # Plot Silhouette Score
    fig, ax = plt.subplots()
    ax.plot(range(2, 9), sil_scores, marker='o')
    ax.set_title('Silhouette Score vs Number of Clusters')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    st.pyplot(fig)

    st.subheader("📊 Cluster Statistics")

# Gabungkan label cluster ke data asli
    st.subheader("📊 Cluster Statistics")

    # Tampilkan statistik deskriptif per cluster
    cluster_summary = df.groupby("Cluster")[[
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]].agg(['mean', 'median', 'min', 'max', 'std'])

    st.dataframe(cluster_summary)

    st.subheader("📉 Boxplot Comparison per Cluster")

    for col in ['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']:
        fig, ax = plt.subplots()
        sns.boxplot(x='Cluster', y=col, data=df, ax=ax)
        ax.set_title(f'{col} by Cluster')
        st.pyplot(fig)

    # Cluster Assignment
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_kmeans)

    st.subheader("Cluster Distribution")
    st.bar_chart(df['Cluster'].value_counts())

# 4. Naive Bayes Classification
elif menu == "🧠 Naive Bayes Classification":
    st.header("🧠 Classify: Is it a Hacker Group Attack?")

    # One-hot Encoding
    categorical_cols = ['Country', 'Target Industry', 'Attack Source',
                        'Security Vulnerability Type', 'Defense Mechanism Used']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    if 'Attack Source_Hacker Group' not in df_encoded.columns:
        st.error("❌ Kolom 'Attack Source_Hacker Group' tidak ditemukan. Pastikan datanya benar.")
    else:
        df_encoded['Is_Hacker_Group_Attack'] = df_encoded['Attack Source_Hacker Group'].astype(int)

        attack_source_cols_onehot = [col for col in df_encoded.columns if 'Attack Source_' in col]
        binary_target_cols_to_drop = [col for col in df_encoded.columns if col.startswith('Is_') and col.endswith('_Attack') and col != 'Is_Hacker_Group_Attack']

        X = df_encoded.drop(columns=['Attack Type', 'Financial Loss (in Million $)', 
                                    'Is_Hacker_Group_Attack'] + attack_source_cols_onehot + binary_target_cols_to_drop)
        y = df_encoded['Is_Hacker_Group_Attack']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("📈 Accuracy", f"{acc:.2%}")

        # st.subheader("Classification Report")
        # st.text(classification_report(y_test, y_pred, target_names=['Not Hacker Group', 'Hacker Group']))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Hacker Group', 'Hacker Group'],
                    yticklabels=['Not Hacker Group', 'Hacker Group'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
