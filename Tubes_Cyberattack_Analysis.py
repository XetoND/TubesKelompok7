import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==============================================================================
# KONFIGURASI HALAMAN & FUNGSI UTAMA
# ==============================================================================

st.set_page_config(layout="wide", page_title="Cybersecurity Dashboard")

@st.cache_data
def load_data():
    """Memuat dataset utama dari file CSV."""
    try:
        df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')
        return df
    except FileNotFoundError:
        st.error("File 'Global_Cybersecurity_Threats_2015-2024.csv' tidak ditemukan.")
        return None

@st.cache_resource
def run_kmeans_simulation(_df):
    """
    Menjalankan simulasi K-Means lengkap: agregasi, scaling, clustering, PCA, dan perhitungan Elbow.
    """
    df_kmeans = _df.copy()
    country_features = df_kmeans.groupby('Country').agg(
        total_incidents=('Country', 'count'),
        avg_financial_loss=('Financial Loss (in Million $)', 'mean'),
        total_affected_users=('Number of Affected Users', 'sum'),
        avg_resolution_time=('Incident Resolution Time (in Hours)', 'mean')
    ).reset_index()
    
    features_to_scale = country_features.drop('Country', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_scale)
    
    # === TAMBAHAN: Menghitung inersia untuk Elbow Method ===
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans_loop = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_loop.fit(scaled_features)
        inertia.append(kmeans_loop.inertia_)
    elbow_df = pd.DataFrame({'k': list(k_range), 'inertia': inertia})
    
    # Melatih model final dengan k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    country_features['cluster'] = kmeans.labels_
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = kmeans.labels_
    pca_df['Country'] = country_features['Country']
    
    return country_features, pca_df, kmeans.cluster_centers_, pca, elbow_df

@st.cache_resource
def train_impact_model(_df):
    """Melatih model klasifikasi RandomForest untuk prediksi dampak."""
    df_impact = _df.copy()
    median_loss = df_impact['Financial Loss (in Million $)'].median()
    df_impact['Impact_Class'] = (df_impact['Financial Loss (in Million $)'] > median_loss).astype(int)
    
    y = df_impact['Impact_Class']
    X = df_impact.drop(['Impact_Class', 'Year', 'Financial Loss (in Million $)'], axis=1)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    encoded_columns = X_encoded.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    best_params = {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'n_estimators': 100}
    rf_model = RandomForestClassifier(random_state=42, **best_params)
    rf_model.fit(X_scaled, y)
    
    return rf_model, scaler, encoded_columns, X.columns

# Memuat data dan menjalankan pemodelan di awal
df = load_data()
if df is not None:
    country_features_df, pca_df, centroids, pca_model, elbow_df = run_kmeans_simulation(df)
    impact_model, impact_scaler, encoded_cols, original_cols = train_impact_model(df)
else:
    st.stop()
    
# ==============================================================================
# STRUKTUR TAMPILAN DASHBOARD
# ==============================================================================

st.sidebar.title("Navigasi 🗺️")
page = st.sidebar.radio("Pilih Halaman:", [
    "Ringkasan Global & Profil Risiko",
    "Visualisasi Data Awal (EDA)",
    "Analisis Pola Ancaman", 
    "Simulasi Prediksi Dampak"
])

# --- Halaman 1: Ringkasan Global & Profil Risiko ---
if page == "Ringkasan Global & Profil Risiko":
    st.title("Dashboard Analisis Ancaman Siber Global 🛡️")
    st.markdown("Halaman ini menyajikan gambaran umum lanskap ancaman siber dan hasil pengelompokan negara berdasarkan profil risiko.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Insiden Tercatat", f"{df.shape[0]:,}")
    col2.metric("Total Kerugian Finansial", f"${int(df['Financial Loss (in Million $)'].sum())} Juta")
    col3.metric("Jumlah Negara Dianalisis", df['Country'].nunique())
    col4.metric("Rata-rata Waktu Resolusi", f"{df['Incident Resolution Time (in Hours)'].mean():.1f} Jam")
    
    st.markdown("---")
    
    st.header("Analisis Hasil K-Means Clustering")

    # === TAMBAHAN: Expander untuk menampilkan Elbow Method ===
    with st.expander("Lihat Proses Teknis: Penentuan K-Optimal dengan Elbow Method"):
        st.markdown("""
        Untuk menentukan jumlah kluster yang optimal, kami menggunakan Elbow Method. Metode ini mencari "titik siku" (elbow point) 
        di mana penambahan jumlah kluster tidak lagi memberikan penurunan inersia (keragaman dalam kluster) yang signifikan.
        Berdasarkan grafik di bawah, kami memilih **k=4** sebagai jumlah kluster yang optimal.
        """)
        fig_elbow = px.line(elbow_df, x='k', y='inertia', markers=True, title='Elbow Method untuk K-Optimal')
        fig_elbow.update_xaxes(title_text='Jumlah Kluster (k)')
        fig_elbow.update_yaxes(title_text='Inersia')
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Menggunakan tab untuk merapikan tampilan
    tab1, tab2 = st.tabs(["Visualisasi Scatter Plot (PCA)", "Visualisasi Peta Geografis"])

    with tab1:
        st.subheader("Visualisasi Kluster dalam 2D (Scatter Plot)")
        st.markdown("Data 4 dimensi hasil agregasi direduksi menjadi 2 dimensi (PC1 & PC2) menggunakan PCA untuk visualisasi.")
        
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='cluster', hover_data=['Country'],
            title='Hasil K-Means Clustering (dalam Ruang 2D PCA)'
        )
        
        centroids_pca = pca_model.transform(centroids)
        fig.add_scatter(
            x=centroids_pca[:, 0], y=centroids_pca[:, 1], mode='markers',
            marker=dict(symbol='x', color='black', size=12), name='Pusat Kluster'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Peta Dunia Profil Risiko Siber")
        # ... (Kode Peta sama seperti sebelumnya) ...
        country_coords = {
            'USA': [37.0902, -95.7129], 'UK': [55.3781, -3.4360], 'Russia': [61.5240, 105.3188],
            'Germany': [51.1657, 10.4515], 'India': [20.5937, 78.9629], 'Japan': [36.2048, 138.2529],
            'China': [35.8617, 104.1954], 'Australia': [ -25.2744, 133.7751], 'Brazil': [-14.2350, -51.9253],
            'Canada': [56.1304, -106.3468]
        }
        map_df = country_features_df.copy()
        map_df['lat'] = map_df['Country'].map(lambda x: country_coords.get(x, [0,0])[0])
        map_df['lon'] = map_df['Country'].map(lambda x: country_coords.get(x, [0,0])[1])
        colors = {0: [255, 0, 0, 160], 1: [0, 255, 0, 160], 2: [0, 0, 255, 160], 3: [255, 255, 0, 160]}
        map_df['color'] = map_df['cluster'].map(colors)
        view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1)
        layer = pdk.Layer('ScatterplotLayer', data=map_df, get_position='[lon, lat]', get_color='color', get_radius=200000, pickable=True)
        tooltip = {"html": "<b>Negara:</b> {Country}<br/><b>Kluster:</b> {cluster}"}
        r = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9', initial_view_state=view_state, layers=[layer], tooltip=tooltip)
        st.pydeck_chart(r)

    st.markdown("---")
    st.subheader("Detail Karakteristik Kluster")
    # ... (Kode Detail Karakteristik Kluster sama seperti sebelumnya) ...
    cluster_choice = st.selectbox("Pilih Kluster untuk dianalisis:", options=sorted(country_features_df['cluster'].unique()))
    st.write(f"**Negara dalam Kluster {cluster_choice}:**")
    countries_in_cluster = country_features_df[country_features_df['cluster'] == cluster_choice]['Country'].tolist()
    st.write(", ".join(countries_in_cluster))
    st.write(f"**Rata-rata Karakteristik Kluster {cluster_choice}:**")
    cluster_details = country_features_df[country_features_df['cluster'] == cluster_choice].drop(['Country', 'cluster'], axis=1).mean()
    st.dataframe(cluster_details)


# --- Halaman lainnya tidak berubah ---
elif page == "Visualisasi Data Awal (EDA)":
    # ... (Kode halaman ini sama seperti sebelumnya) ...
    st.title("Visualisasi Data Awal (Exploratory Data Analysis) 📊")
    st.markdown("Pilih fitur dari dataset untuk melihat distribusinya.")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    feature_type = st.radio("Pilih tipe fitur:", ("Kategorikal", "Numerik"))
    if feature_type == "Kategorikal":
        selected_feature = st.selectbox("Pilih Fitur Kategorikal:", categorical_cols)
        st.subheader(f"Distribusi Fitur: {selected_feature}")
        feature_counts = df[selected_feature].value_counts()
        fig = px.bar(feature_counts, x=feature_counts.index, y=feature_counts.values, labels={'x': selected_feature, 'y': 'Jumlah'})
        st.plotly_chart(fig, use_container_width=True)
    elif feature_type == "Numerik":
        selected_feature = st.selectbox("Pilih Fitur Numerik:", numeric_cols)
        st.subheader(f"Distribusi Fitur: {selected_feature}")
        fig = px.histogram(df, x=selected_feature, marginal="box", title=f'Distribusi {selected_feature}')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Analisis Pola Ancaman":
    # ... (Kode halaman ini sama seperti sebelumnya) ...
    st.title("Analisis Mendalam Pola Ancaman 🔎")
    st.markdown("Gunakan filter di sidebar untuk menjelajahi data secara interaktif.")
    st.sidebar.header("Filter Analisis")
    selected_countries = st.sidebar.multiselect("Pilih Negara:", options=sorted(df['Country'].unique()), default=sorted(df['Country'].unique()))
    selected_industries = st.sidebar.multiselect("Pilih Industri:", options=sorted(df['Target Industry'].unique()), default=sorted(df['Target Industry'].unique()))
    year_range = st.sidebar.slider("Pilih Rentang Tahun:", df['Year'].min(), df['Year'].max(), (df['Year'].min(), df['Year'].max()))
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Target Industry'].isin(selected_industries)) &
        (df['Year'].between(year_range[0], year_range[1]))
    ]
    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Tipe Serangan")
            attack_counts = filtered_df['Attack Type'].value_counts().head(5)
            fig1 = px.bar(attack_counts, y=attack_counts.index, x=attack_counts.values, orientation='h', labels={'y': 'Tipe Serangan', 'x': 'Jumlah'})
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("Top 5 Industri Target")
            industry_counts = filtered_df['Target Industry'].value_counts().head(5)
            fig2 = px.bar(industry_counts, y=industry_counts.index, x=industry_counts.values, orientation='h', labels={'y': 'Industri', 'x': 'Jumlah'})
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

elif page == "Simulasi Prediksi Dampak":
    # ... (Kode halaman ini sama seperti sebelumnya) ...
    st.title("Simulasi Prediksi Dampak Finansial 💸")
    st.markdown("Gunakan model Machine Learning terbaik kita untuk memprediksi apakah sebuah serangan berpotensi berdampak tinggi.")
    with st.expander("Lihat Detail Performa Model"):
        st.info("Akurasi Model (Random Forest): **52.17%**. Model ini memberikan prediksi yang sedikit lebih baik dari tebakan acak.")
        importances = impact_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': encoded_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.write("**Fitur Paling Berpengaruh Terhadap Prediksi:**")
        st.dataframe(feature_importance_df.head(5))
    st.sidebar.header("Input untuk Simulasi")
    with st.sidebar.form(key='simulation_form'):
        categorical_features_for_sim = {
            'Country': sorted(df['Country'].unique()),
            'Attack Type': sorted(df['Attack Type'].unique()),
            'Target Industry': sorted(df['Target Industry'].unique()),
            'Attack Source': sorted(df['Attack Source'].unique()),
            'Security Vulnerability Type': sorted(df['Security Vulnerability Type'].unique()),
            'Defense Mechanism Used': sorted(df['Defense Mechanism Used'].unique())
        }
        input_dict = {}
        for feature, options in categorical_features_for_sim.items():
            input_dict[feature] = st.selectbox(f"{feature}:", options)
        input_dict['Number of Affected Users'] = st.number_input("Jumlah Pengguna Terdampak:", min_value=0, value=500000)
        input_dict['Incident Resolution Time (in Hours)'] = st.number_input("Waktu Resolusi (Jam):", min_value=1, value=36)
        submit_button = st.form_submit_button(label='Prediksi Dampak')
    if submit_button:
        input_df = pd.DataFrame([input_dict], columns=original_cols)
        input_encoded = pd.get_dummies(input_df, columns=input_df.select_dtypes(include=['object']).columns)
        input_reindexed = input_encoded.reindex(columns=encoded_cols, fill_value=0)
        input_scaled = impact_scaler.transform(input_reindexed)
        prediction = impact_model.predict(input_scaled)
        prediction_proba = impact_model.predict_proba(input_scaled)
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.error("Prediksi: **HIGH IMPACT**")
        else:
            st.success("Prediksi: **LOW IMPACT**")
        st.write("Tingkat Keyakinan Model:")
        st.progress(prediction_proba[0][prediction[0]])
        st.write(f"Probabilitas Low Impact: {prediction_proba[0][0]:.2%}")
        st.write(f"High Impact: {prediction_proba[0][1]:.2%}")
