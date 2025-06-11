import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix

# --- Konfigurasi Halaman Streamlit ---
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

    # 1. Preprocessing
    df_kmeans = df.copy()
    df_kmeans = pd.get_dummies(df_kmeans, drop_first=True)

    numeric_cols = [
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]
    scaler = StandardScaler()
    df_kmeans[numeric_cols] = scaler.fit_transform(df_kmeans[numeric_cols])

    # 2. Cari k optimal dengan Silhouette Score
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(df_kmeans)
        sil_scores.append(silhouette_score(df_kmeans, clusters))

    optimal_k = range(2, 9)[sil_scores.index(max(sil_scores))]
    st.write(f"✅ **Optimal Number of Clusters:** {optimal_k}")

    fig, ax = plt.subplots()
    ax.plot(range(2, 9), sil_scores, marker='o')
    ax.set_title('Silhouette Score vs Number of Clusters')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    st.pyplot(fig)

    # 3. Clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(df_kmeans)

    st.subheader("🔢 Cluster Distribution")
    st.bar_chart(df_clustered['Cluster'].value_counts())

    # 4. Statistik per Cluster
    st.subheader("📊 Cluster Statistics")
    cluster_summary = df_clustered.groupby("Cluster")[numeric_cols].agg(
        ['mean', 'median', 'min', 'max', 'std']
    )
    st.dataframe(cluster_summary)

    # 5. Boxplot Visualisasi
    st.subheader("📉 Boxplot Comparison per Cluster")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x='Cluster', y=col, data=df_clustered, ax=ax)
        ax.set_title(f'{col} by Cluster')
        st.pyplot(fig)


    # --- 3. Tampilan Klasifikasi dengan Regresi Logistik ---
    elif menu == "🧠 Klasifikasi dengan Regresi Logistik":
        st.header("🧠 Deteksi Ransomware dengan Regresi Logistik (Supervised)")
        st.write("Melatih model untuk memprediksi apakah sebuah serangan termasuk kategori 'Ransomware'.")

        df_logreg = df_initial.copy()
        
        # Membuat target biner: 1 jika 'Ransomware', 0 jika bukan
        df_logreg['Is_Ransomware'] = (df_logreg['Attack Type'] == 'Ransomware').astype(int)

        # Memilih fitur dan target
        features = ['Target Industry', 'Financial Loss (in Million $)', 'Number of Affected Users', 'Security Vulnerability Type']
        target = 'Is_Ransomware'

        X = pd.get_dummies(df_logreg[features], drop_first=True)
        y = df_logreg[target]

        # Membagi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # Scaling fitur numerik
        numerical_cols = ['Financial Loss (in Million $)', 'Number of Affected Users']
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        # --- PERUBAHAN: Menghapus tombol dan menjalankan analisis secara otomatis ---
        with st.spinner("Model sedang dilatih dan dievaluasi..."):
            # Menggunakan class_weight='balanced' untuk menangani data tidak seimbang
            model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.success("🎉 Model berhasil dilatih dan dievaluasi!")

            # Menampilkan hasil evaluasi
            st.subheader("Hasil Evaluasi Model")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Akurasi Model", value=f"{accuracy:.2%}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Laporan Klasifikasi")
                report = classification_report(y_test, y_pred, target_names=['Bukan Ransomware', 'Ransomware'])
                st.text_area("Classification Report", report, height=250)
            
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Prediksi Bukan', 'Prediksi Ransomware'],
                            yticklabels=['Aktual Bukan', 'Aktual Ransomware'], ax=ax_cm)
                ax_cm.set_xlabel('Prediksi')
                ax_cm.set_ylabel('Aktual')
                st.pyplot(fig_cm)
