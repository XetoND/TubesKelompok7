import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score

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
    st.write("Melatih model untuk memprediksi apakah sebuah serangan termasuk kategori 'Ransomware' atau bukan, lengkap dengan analisis statistik.")

    # --- 1. PREPROCESSING DATA ---
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

    # Membuat variabel dummy untuk fitur kategoris
    X = pd.get_dummies(df_logreg[features], drop_first=True)
    y = df_logreg[target]
    
    # Membagi data latih dan data uji SEBELUM scaling dan menambah konstanta
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Membuat salinan eksplisit untuk menghindari SettingWithCopyWarning
    X_train = X_train_orig.copy()
    X_test = X_test_orig.copy()

    # Scaling fitur numerik
    numerical_cols_logreg = ['Financial Loss (in Million $)', 'Number of Affected Users']
    scaler_logreg = StandardScaler()
    
    X_train[numerical_cols_logreg] = scaler_logreg.fit_transform(X_train[numerical_cols_logreg])
    X_test[numerical_cols_logreg] = scaler_logreg.transform(X_test[numerical_cols_logreg])
    
    # Menambahkan konstanta/intercept untuk model statsmodels
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    # --- 2. PEMODELAN DAN ANALISIS STATISTIK DENGAN STATSMODELS ---
    st.subheader("Analisis Statistik Model")
    with st.spinner("Melatih model statistik dan menghasilkan ringkasan..."):
        # Membuat dan melatih model dengan statsmodels
        logit_model = sm.Logit(y_train, X_train_sm)
        result = logit_model.fit()
        st.success("✅ Model statistik berhasil dilatih!")

        # Menampilkan ringkasan model
        st.text_area("Ringkasan Model Statistik (dari Statsmodels)", result.summary().as_text(), height=450)

    # --- INTERPRETASI SESUAI RUBRIK PENILAIAN ---
    st.subheader("Interpretasi Hasil Sesuai Rubrik Penilaian")

    # 1. Interpretasi Koefisien Model
    with st.expander("🔹 Interpretasi Koefisien Model"):
        st.markdown("""
        Koefisien (kolom `coef` pada tabel di atas) menunjukkan perubahan dalam **log-odds** dari target (`Is_Ransomware`) untuk setiap kenaikan satu unit pada variabel prediktor, dengan asumsi variabel lain konstan.
        - **Variabel Kontinu** (contoh: `Financial Loss`): Jika koefisien positif, semakin besar kerugian finansial, semakin tinggi kemungkinan serangan tersebut adalah Ransomware.
        - **Variabel Diskrit/Kategoris** (contoh: `Target Industry_Healthcare`): Nilai koefisien membandingkan *log-odds* dari industri tersebut dengan industri referensi. Nilai positif berarti industri tersebut lebih mungkin mengalami serangan Ransomware dibandingkan industri referensi.
        
        Untuk interpretasi yang lebih intuitif, kita gunakan **Odds Ratio** (`exp(coef)`).
        """)
        odds_ratios = pd.DataFrame(np.exp(result.params), columns=['Odds Ratio'])
        st.dataframe(odds_ratios)
        st.markdown("""
        - **Odds Ratio > 1**: Meningkatkan kemungkinan terjadinya Ransomware.
        - **Odds Ratio < 1**: Menurunkan kemungkinan terjadinya Ransomware.
        - **Odds Ratio = 1**: Tidak ada pengaruh.
        """)
    
    # 2. Variabel Signifikan berdasarkan P-Value
    with st.expander("🔹 Variabel Signifikan (berdasarkan P-value)"):
        st.markdown("""
        Kolom **`P>|z|`** pada tabel ringkasan menunjukkan **p-value**. Nilai ini menguji signifikansi statistik dari setiap variabel.
        - **P-value < 0.05**: Variabel dianggap **signifikan secara statistik**. Artinya, variabel tersebut memiliki pengaruh yang nyata terhadap kemungkinan sebuah serangan adalah Ransomware.
        - **P-value >= 0.05**: Variabel dianggap **tidak signifikan**. Artinya, tidak ada cukup bukti statistik untuk menyatakan variabel tersebut mempengaruhi prediksi.
        
        Dari tabel di atas, kita dapat mencari variabel dengan `P>|z| < 0.05` untuk mengetahui prediktor yang paling berpengaruh.
        """)

    # 3. Goodness of Fit Model
    with st.expander("🔹 Goodness of Fit (Kebaikan Model)"):
        st.markdown(f"""
        *Goodness of fit* menunjukkan seberapa baik model cocok dengan data observasi.
        
        **1. Pseudo R-squared**
        - Nilai **Pseudo R-squ.** dari model ini adalah **{result.prsquared:.4f}**.
        - Nilai ini mengindikasikan bahwa sekitar **{result.prsquared:.2%}** dari variabilitas dalam variabel target (apakah serangan itu Ransomware atau bukan) dapat dijelaskan oleh model. Semakin tinggi nilainya (mendekati 1), semakin baik.

        **2. Kurva ROC-AUC**
        - Kurva ROC memvisualisasikan kemampuan model dalam membedakan antara kelas positif (Ransomware) dan negatif. **AUC (Area Under the Curve)** yang mendekati 1 menunjukkan performa yang sangat baik.
        """)
        
        y_pred_proba = result.predict(X_test_sm) # Prediksi probabilitas pada data test
        auc_score = roc_auc_score(y_test, y_pred_proba)
        st.metric(label="**ROC-AUC Score**", value=f"**{auc_score:.4f}**")
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--', label='Garis Referensi (Acak)')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # 4. Interpretasi Hasil Evaluasi
    with st.expander("🔹 Interpretasi Hasil Evaluasi (Confusion Matrix & Metrik Lainnya)"):
        y_pred = (y_pred_proba > 0.5).astype(int) # Konversi probabilitas ke kelas biner
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric(label="**Akurasi Model**", value=f"**{accuracy:.2%}**")
        st.markdown(f"Secara keseluruhan, **{accuracy:.2%}** prediksi model pada data uji benar.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Laporan Klasifikasi")
            report_dict = classification_report(y_test, y_pred, target_names=['Bukan Ransomware', 'Ransomware'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df)

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Prediksi Bukan', 'Prediksi Ransomware'],
                        yticklabels=['Aktual Bukan', 'Aktual Ransomware'], ax=ax_cm)
            ax_cm.set_xlabel('Prediksi Model')
            ax_cm.set_ylabel('Nilai Aktual')
            st.pyplot(fig_cm)
        
        st.markdown(f"""
        - **Presisi (Precision)**:
          - Untuk kelas `Ransomware` ({report_dict['Ransomware']['precision']:.2f}): Dari semua yang diprediksi sebagai Ransomware, {report_dict['Ransomware']['precision']:.2%} di antaranya benar.
        
        - **Recall (Sensitivity)**:
          - Untuk kelas `Ransomware` ({report_dict['Ransomware']['recall']:.2f}): Dari semua kasus Ransomware yang sebenarnya, model berhasil mengidentifikasi {report_dict['Ransomware']['recall']:.2%} di antaranya. Ini adalah metrik penting jika tujuannya adalah menangkap sebanyak mungkin kasus Ransomware.
          
        - **F1-Score**: Rata-rata harmonik dari Presisi dan Recall, memberikan gambaran performa yang seimbang.
        """)
