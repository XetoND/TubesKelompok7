import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Ancaman Ransomware",
    page_icon="🛡️",
    layout="wide"
)

# --- Fungsi untuk Melatih dan Mengevaluasi Model ---
def train_and_evaluate(df):
    """
    Fungsi untuk melakukan pra-pemrosesan data, melatih model regresi logistik,
    dan mengevaluasi kinerjanya.
    """
    # Menghapus baris dengan nilai yang hilang
    df.dropna(inplace=True)

    # --- Mendefinisikan Ulang Target Prediksi ---
    # Fokus pada deteksi 'Ransomware'
    df['Is_Ransomware'] = (df['Attack Type'] == 'Ransomware').astype(int)

    # Memilih fitur dan target baru
    features = [
        'Target Industry',
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Security Vulnerability Type'
    ]
    target = 'Is_Ransomware'

    # --- Pra-pemrosesan Data ---
    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Scaling Fitur Numerik ---
    numerical_cols = ['Financial Loss (in Million $)', 'Number of Affected Users']
    # Memastikan kolom numerik ada di X_train sebelum scaling
    cols_to_scale = [col for col in numerical_cols if col in X_train.columns]

    if cols_to_scale:
        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # --- Implementasi Model Regresi Logistik ---
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # Memprediksi hasil pada data uji
    y_pred = model.predict(X_test)

    # Mengembalikan hasil evaluasi
    return y_test, y_pred

# --- Antarmuka Aplikasi Streamlit ---

st.title("🛡️ Aplikasi Deteksi Ancaman Ransomware")
st.write("""
Aplikasi ini menggunakan model **Regresi Logistik** untuk memprediksi apakah sebuah ancaman siber
merupakan serangan **Ransomware** berdasarkan beberapa fitur. Unggah file CSV Anda untuk memulai.
""")

# --- Sidebar untuk Unggah File dan Informasi ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])
    st.info("""
    **Catatan:** Pastikan file CSV Anda memiliki kolom berikut:
    - `Attack Type`
    - `Target Industry`
    - `Financial Loss (in Million $)`
    - `Number of Affected Users`
    - `Security Vulnerability Type`
    """)

if uploaded_file is not None:
    # --- Membaca dan Menampilkan Data ---
    try:
        df = pd.read_csv(uploaded_file)
        st.header("📊 Pratinjau Data")
        st.dataframe(df.head())

        # --- Tombol untuk Memulai Analisis ---
        if st.button("🚀 Latih Model dan Lakukan Prediksi", type="primary"):
            with st.spinner('Model sedang dilatih... Mohon tunggu...'):
                y_test, y_pred = train_and_evaluate(df.copy()) # Menggunakan salinan data

                st.success("🎉 Model berhasil dilatih dan dievaluasi!")

                # --- Menampilkan Hasil Evaluasi ---
                st.header("### Hasil Evaluasi Model Regresi Logistik ###")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Akurasi Model")
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric(label="Akurasi", value=f"{accuracy:.2%}")

                    st.subheader("Laporan Klasifikasi")
                    report = classification_report(y_test, y_pred, target_names=['Bukan Ransomware (0)', 'Ransomware (1)'], output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

                with col2:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Bukan Ransomware', 'Ransomware'],
                                yticklabels=['Bukan Ransomware', 'Ransomware'], ax=ax)
                    ax.set_xlabel('Prediksi Model')
                    ax.set_ylabel('Kenyataan (Aktual)')
                    ax.set_title('Confusion Matrix - Deteksi Ransomware')
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan unggah file CSV melalui sidebar untuk memulai.")
