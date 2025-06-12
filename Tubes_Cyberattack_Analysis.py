import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

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

    # Validasi DataFrame
    if df.empty:
        st.error("DataFrame kosong. Pastikan data telah dimuat dengan benar.")
    else:
        # Memastikan kolom yang diperlukan ada di DataFrame
        required_columns = [
            'Attack Type', 'Target Industry', 'Financial Loss (in Million $)', 
            'Number of Affected Users', 'Security Vulnerability Type'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Kolom berikut tidak ditemukan di dataset: {missing_columns}. Harap periksa nama kolom atau dataset Anda.")
        else:
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

            df_logreg_cleaned = df_logreg.dropna(subset=features + [target])

            X = pd.get_dummies(df_logreg_cleaned[features], drop_first=True)
            y = df_logreg_cleaned[target]

            if len(y.unique()) < 2:
                st.warning("Target variabel 'Is_Ransomware' hanya memiliki satu kelas setelah pembersihan data. Tidak dapat melakukan klasifikasi.")
            elif len(X) == 0:
                st.warning("Tidak ada data yang tersisa setelah pembersihan dan one-hot encoding. Silakan periksa data Anda.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

                numerical_cols_logreg = ['Financial Loss (in Million $)', 'Number of Affected Users']
                numerical_cols_present = [col for col in numerical_cols_logreg if col in X_train.columns]

                scaler_logreg = StandardScaler()
                if numerical_cols_present:
                    X_train[numerical_cols_present] = scaler_logreg.fit_transform(X_train[numerical_cols_present])
                    X_test[numerical_cols_present] = scaler_logreg.transform(X_test[numerical_cols_present])
                else:
                    st.info("Tidak ada kolom numerik yang terdeteksi untuk scaling dalam model Regresi Logistik.")

                if y_train.value_counts().min() == 0:
                    st.warning("Salah satu kelas dalam target training set tidak memiliki sampel. SMOTE tidak dapat diterapkan.")
                    X_train_resampled, y_train_resampled = X_train, y_train 
                else:
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                with st.spinner("Model sedang dilatih dan dievaluasi..."):
                    # Scikit-learn model for prediction and metrics
                    model_sklearn = LogisticRegression(random_state=42, max_iter=1000)
                    model_sklearn.fit(X_train_resampled, y_train_resampled)
                    y_pred = model_sklearn.predict(X_test)

                    st.success("🎉 Model berhasil dilatih dan dievaluasi!")

                    # Menampilkan hasil evaluasi (dari sklearn model)
                    st.subheader("Hasil Evaluasi Model")
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric(label="Akurasi Model", value=f"{accuracy:.2%}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Laporan Klasifikasi")
                        report = classification_report(y_test, y_pred, target_names=['Bukan Ransomware', 'Ransomware'], output_dict=False, zero_division='warn')
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
                        plt.close(fig_cm) 

                    # Menampilkan interpretasi koefisien model (dari sklearn model)
                    st.subheader("Interpretasi Koefisien")
                    coef_sklearn = pd.DataFrame({
                        'Feature': X.columns,
                        'Coefficient': model_sklearn.coef_[0]
                    })
                    coef_sklearn['Abs_Coefficient'] = coef_sklearn['Coefficient'].abs()
                    coef_sklearn = coef_sklearn.sort_values(by='Abs_Coefficient', ascending=False)
                    st.write(coef_sklearn[['Feature', 'Coefficient']])
                    st.write("Koefisien positif menunjukkan peningkatan kemungkinan serangan sebagai Ransomware, sedangkan koefisien negatif menunjukkan penurunan kemungkinan.")

                    # # --- Variabel Signifikan (p-value) menggunakan Statsmodels ---
                    # st.subheader("Variabel Signifikan (p-value dari Statsmodels)")
                    # st.write("Untuk analisis statistik inferensial dan p-value, kita akan menggunakan library `statsmodels`.")
                    
                    # # --- Robustness checks before statsmodels ---
                    # if X_train_resampled.empty or y_train_resampled.empty:
                    #     st.warning("Data latih (X_train_resampled atau y_train_resampled) kosong setelah resampling. Tidak dapat menghitung p-value.")
                    # elif X_train_resampled.isnull().values.any() or X_train_resampled.isin([float('inf'), float('-inf')]).values.any():
                    #     st.warning("Data latih (X_train_resampled) mengandung nilai NaN atau Inf. Harap periksa dan bersihkan data Anda.")
                    # elif y_train_resampled.isnull().any():
                    #     st.warning("Target latih (y_train_resampled) mengandung nilai NaN. Harap periksa dan bersihkan data Anda.")
                    # elif len(y_train_resampled.unique()) < 2:
                    #     st.warning("Target latih (y_train_resampled) hanya memiliki satu kelas. Tidak dapat melakukan regresi logistik dengan statsmodels karena membutuhkan setidaknya dua kelas.")
                    # else:
                    #     # Identify columns that are constant after SMOTE, these can cause issues
                    #     constant_columns = X_train_resampled.columns[X_train_resampled.nunique() == 1]
                    #     if not constant_columns.empty:
                    #         st.info(f"Kolom berikut memiliki nilai konstan (setelah SMOTE) dan akan dihapus untuk model Statsmodels: {', '.join(constant_columns)}.")
                    #         X_train_resampled_sm = X_train_resampled.drop(columns=constant_columns)
                    #     else:
                    #         X_train_resampled_sm = X_train_resampled.copy()

                    #     # Add a constant (intercept) to the training data for statsmodels
                    #     X_train_sm = sm.add_constant(X_train_resampled_sm, has_constant='add')
                        
                    #     try:
                    #         logit_model = sm.Logit(y_train_resampled, X_train_sm)
                    #         # ***** PERBAIKAN UTAMA DI SINI: Menggunakan fit_regularized() *****
                    #         # Ini lebih tangguh terhadap perfect separation.
                    #         # alpha=1.0 untuk L1 regularization (Lasso), yang bisa menekan koefisien ke nol.
                    #         result = logit_model.fit_regularized(disp=False, maxiter=2000, alpha=1.0) 

                    #         st.text("Ringkasan Model Statsmodels:")
                    #         with st.expander("Lihat Ringkasan Lengkap Model Statsmodels"):
                    #             st.code(result.summary().as_text(), language='text')

                    #         # Extract p-values (Note: p-values from regularized models should be interpreted with caution)
                    #         # For L1 regularization, some p-values might be NaN if coefficients are shrunk to zero.
                    #         p_values_df = result.pvalues.to_frame(name='P-value')
                    #         p_values_df.index.name = 'Feature'
                    #         p_values_df = p_values_df.reset_index()
                            
                    #         # Handle potential NaN p-values from regularization by filling them with 1.0 (non-significant)
                    #         p_values_df['P-value'] = p_values_df['P-value'].fillna(1.0) 

                    #         p_values_df['Significance'] = p_values_df['P-value'].apply(lambda p: 'Signifikan (<0.05)' if p < 0.05 else 'Tidak Signifikan (>=0.05)')
                            
                    #         p_values_df = p_values_df.sort_values(by='P-value')

                    #         st.write("Variabel berdasarkan P-value:")
                    #         st.dataframe(p_values_df)
                    #         st.write("Variabel dengan p-value kurang dari 0.05 (< 0.05) umumnya dianggap signifikan secara statistik, yang berarti perubahan pada variabel tersebut memiliki efek yang signifikan pada kemungkinan serangan sebagai Ransomware.")
                    #         st.info("Catatan: P-value dari model regularisasi (seperti yang digunakan di sini) harus diinterpretasikan dengan hati-hati karena regularisasi itu sendiri memengaruhi estimasi parameter dan standar error.")

                    #     except Exception as e:
                    #         st.error(f"Terjadi kesalahan saat melatih model Statsmodels: {e}")
                    #         st.write("Ini mungkin disebabkan oleh masalah konvergensi, multikolinearitas ekstrem, atau masalah data lainnya yang belum teratasi.")
                    #         st.write("Penggunaan `fit_regularized` seharusnya mengurangi masalah `PerfectSeparationError`, tetapi jika masih terjadi, bisa jadi ada kombinasi fitur yang sangat kompleks atau data yang sangat langka.")
                    #         st.write(f"Detail kesalahan teknis: `{e}`")


                    # Menampilkan Goodness of Fit (dari sklearn model)
                    st.subheader("Goodness of Fit")
                    if 1 in y_test.unique() and 0 in y_test.unique(): 
                        probas = model_sklearn.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, probas)
                        st.metric(label="ROC-AUC Score", value=f"{roc_auc:.2f}")
                        fig_roc, ax_roc = plt.subplots()
                        fpr, tpr, _ = roc_curve(y_test, probas)
                        ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
                        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='grey')
                        ax_roc.set_xlabel("False Positive Rate")
                        ax_roc.set_ylabel("True Positive Rate")
                        ax_roc.legend(loc="best")
                        st.pyplot(fig_roc)
                        plt.close(fig_roc) 
                    else:
                        st.warning("Tidak ada sampel dari kedua kelas (Ransomware dan Bukan Ransomware) di test set, ROC-AUC tidak dapat dihitung.")
    
