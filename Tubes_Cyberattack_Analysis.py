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

# Judul Utama
st.title("🔐 Analisis Ancaman Serangan Siber (2015–2024)")

# Fungsi untuk memuat data dengan cache agar lebih cepat
@st.cache_data
def load_data():
    # Ganti 'Global_Cybersecurity_Threats_2015-2024.csv' dengan path file Anda jika berbeda
    df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')
    return df

df = load_data()

# --- Navigasi Sidebar ---
# Saya mengubah nama menu agar sesuai dengan implementasi (Regresi Logistik)
menu = st.sidebar.radio(
    "Navigasi",
    [
        "📊 Ringkasan Dataset",
        "📈 Visualisasi Data",
        "📌 Clustering K-Means",
        "🧠 Klasifikasi Regresi Logistik"
    ]
)

# --- 1. Halaman Ringkasan Dataset ---
if menu == "📊 Ringkasan Dataset":
    st.header("📊 Ringkasan Dataset")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif (Fitur Numerik)")
    # Menghapus kolom 'Year' dari statistik deskriptif karena merupakan data ordinal
    st.write(df.drop(columns=['Year']).describe())

    st.subheader("Jumlah Nilai Unik (Fitur Kategoris)")
    for col in df.select_dtypes('object').columns:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())

# --- 2. Halaman Visualisasi Data ---
elif menu == "📈 Visualisasi Data":
    st.header("📈 Eksplorasi Visual")

    # Distribusi Jenis Serangan dan Industri Target
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi: Jenis Serangan")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, y='Attack Type', ax=ax1, order=df['Attack Type'].value_counts().index)
        st.pyplot(fig1)

    with col2:
        st.subheader("Distribusi: Industri Target")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, y='Target Industry', ax=ax2, order=df['Target Industry'].value_counts().index)
        st.pyplot(fig2)

    # Histogram untuk fitur numerik
    st.subheader("Distribusi Fitur Numerik")
    numeric_cols = [
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribusi {col}')
        st.pyplot(fig)

# --- 3. Halaman Clustering K-Means ---
elif menu == "📌 Clustering K-Means":
    st.header("📌 Analisis Cluster dengan K-Means (Unsupervised)")
    st.write("Mengelompokkan data serangan siber ke dalam beberapa cluster berdasarkan kesamaan fitur.")

    # 1. Preprocessing
    df_kmeans = df.copy()
    # One-hot encoding untuk kolom kategoris
    df_kmeans = pd.get_dummies(df_kmeans, drop_first=True)

    numeric_cols = [
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]
    # Scaling fitur numerik
    scaler = StandardScaler()
    df_kmeans[numeric_cols] = scaler.fit_transform(df_kmeans[numeric_cols])

    # 2. Mencari jumlah cluster (k) optimal dengan Silhouette Score
    st.subheader("Mencari Jumlah Cluster Optimal (k)")
    with st.spinner("Menghitung Silhouette Score untuk berbagai nilai k..."):
        sil_scores = []
        k_range = range(2, 9)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = km.fit_predict(df_kmeans)
            sil_scores.append(silhouette_score(df_kmeans, clusters))

        # Menemukan k dengan score tertinggi
        optimal_k = k_range[sil_scores.index(max(sil_scores))]
        st.success(f"✅ **Jumlah Cluster Optimal Ditemukan:** {optimal_k}")

        # Plot Silhouette Score
        fig, ax = plt.subplots()
        ax.plot(k_range, sil_scores, marker='o')
        ax.set_title('Silhouette Score vs Jumlah Cluster (k)')
        ax.set_xlabel('Jumlah Cluster (k)')
        ax.set_ylabel('Silhouette Score')
        st.pyplot(fig)

    # 3. Melakukan Clustering dengan k optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(df_kmeans)

    st.subheader("🔢 Distribusi Data per Cluster")
    st.bar_chart(df_clustered['Cluster'].value_counts())

    # 4. Menampilkan statistik untuk setiap cluster
    st.subheader("📊 Statistik Deskriptif per Cluster")
    cluster_summary = df_clustered.groupby("Cluster")[numeric_cols].agg(
        ['mean', 'median', 'min', 'max', 'std']
    )
    st.dataframe(cluster_summary)

    # 5. Visualisasi Boxplot untuk membandingkan cluster
    st.subheader("📉 Perbandingan Fitur Numerik antar Cluster")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x='Cluster', y=col, data=df_clustered, ax=ax)
        ax.set_title(f'Perbandingan {col} berdasarkan Cluster')
        st.pyplot(fig)

# --- 4. Halaman Klasifikasi dengan Regresi Logistik ---
elif menu == "🧠 Klasifikasi Regresi Logistik":
    st.header("🧠 Deteksi Ransomware dengan Regresi Logistik (Supervised)")
    st.write("Melatih model untuk memprediksi apakah sebuah serangan termasuk kategori 'Ransomware' atau bukan.")

    # Menggunakan df.copy() bukan df_initial.copy()
    df_logreg = df.copy()
    
    # Membuat target biner: 1 jika 'Ransomware', 0 jika bukan
    df_logreg['Is_Ransomware'] = (df_logreg['Attack Type'] == 'Ransomware').astype(int)

    # Memilih fitur dan target
    features = [
        'Target Industry',
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Security Vulnerability Type'
    ]
    target = 'Is_Ransomware'

    X = pd.get_dummies(df_logreg[features], drop_first=True)
    y = df_logreg[target]

    # Membagi data latih dan data uji
    # Menggunakan stratify=y untuk memastikan proporsi target sama di data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scaling fitur numerik
    numerical_cols_logreg = ['Financial Loss (in Million $)', 'Number of Affected Users']
    scaler_logreg = StandardScaler()
    X_train[numerical_cols_logreg] = scaler_logreg.fit_transform(X_train[numerical_cols_logreg])
    X_test[numerical_cols_logreg] = scaler_logreg.transform(X_test[numerical_cols_logreg])

    # Melatih model dan menampilkan hasil secara otomatis
    with st.spinner("Model sedang dilatih dan dievaluasi..."):
        # Menggunakan class_weight='balanced' untuk menangani data yang tidak seimbang
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
            report = classification_report(y_test, y_pred, target_names=['Bukan Ransomware', 'Ransomware'], output_dict=False)
            st.text_area("Classification Report", report, height=200)
        
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
