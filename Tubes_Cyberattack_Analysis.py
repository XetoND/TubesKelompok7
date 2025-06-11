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
st.set_page_config(
    page_title="Analisis Ancaman Siber",
    page_icon="🛡️",
    layout="wide"
)

# --- Judul Utama Aplikasi ---
st.title("🛡️ Dasbor Analisis Ancaman Siber")
st.write("Aplikasi ini menggabungkan model Unsupervised (K-Means) dan Supervised (Regresi Logistik) untuk menganalisis data ancaman siber.")

# --- Fungsi untuk Memuat Data dengan Caching ---
@st.cache_data
def load_data():
    """Memuat dataset dari file CSV."""
    try:
        df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        st.error("File 'Global_Cybersecurity_Threats_2015-2024.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None

# Memuat data
df_initial = load_data()

# --- Navigasi di Sidebar ---
st.sidebar.title("Navigasi Analisis")
menu = st.sidebar.radio(
    "Pilih Model atau Tampilan:",
    ["📊 Ringkasan & Visualisasi", "📌 Analisis K-Means Clustering", "🧠 Klasifikasi dengan Regresi Logistik"]
)

if df_initial is not None:
    # --- 1. Tampilan Ringkasan & Visualisasi ---
    if menu == "📊 Ringkasan & Visualisasi":
        st.header("📊 Ringkasan dan Visualisasi Data")
        
        st.subheader("Pratinjau Dataset")
        st.dataframe(df_initial.head())

        st.subheader("Statistik Deskriptif")
        st.write(df_initial.describe())

        st.subheader("Visualisasi Distribusi Data")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df_initial, y='Attack Type', ax=ax1, order=df_initial['Attack Type'].value_counts().index)
            ax1.set_title("Distribusi Tipe Serangan")
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df_initial, y='Target Industry', ax=ax2, order=df_initial['Target Industry'].value_counts().index)
            ax2.set_title("Distribusi Industri Target")
            st.pyplot(fig2)

    # --- 2. Tampilan Analisis K-Means Clustering ---
    elif menu == "📌 Analisis K-Means Clustering":
        st.header("📌 Analisis K-Means Clustering (Unsupervised)")
        st.write("Mengelompokkan serangan siber ke dalam beberapa cluster berdasarkan kesamaan fitur.")

        df_kmeans = df_initial.copy()
        df_processed = pd.get_dummies(df_kmeans, drop_first=True)

        scaler = StandardScaler()
        df_processed_scaled = scaler.fit_transform(df_processed)

        st.subheader("Menentukan Jumlah Cluster Optimal (k)")
        with st.spinner("Menghitung Silhouette Score..."):
            k_range = range(2, 9)
            sil_scores = [silhouette_score(df_processed_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(df_processed_scaled)) for k in k_range]
        
        optimal_k = k_range[sil_scores.index(max(sil_scores))]
        st.success(f"✅ **Jumlah Cluster Optimal (ditemukan dengan Silhouette Score):** `{optimal_k}`")

        fig, ax = plt.subplots()
        ax.plot(k_range, sil_scores, marker='o')
        ax.set_title('Silhouette Score vs. Jumlah Cluster (k)')
        ax.set_xlabel('Jumlah Cluster (k)')
        ax.set_ylabel('Silhouette Score')
        st.pyplot(fig)

        st.subheader("Hasil Clustering")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_initial['Cluster'] = kmeans.fit_predict(df_processed_scaled)
        st.bar_chart(df_initial['Cluster'].value_counts())
        st.write("Data dengan Label Cluster:")
        st.dataframe(df_initial.head())

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
