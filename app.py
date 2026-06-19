import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Data Siswa",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Data Siswa")

# Upload File
uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df)

    # Informasi Dataset
    st.subheader("Informasi Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", len(df))
    col2.metric("Jumlah Kolom", len(df.columns))
    col3.metric("Jumlah Kelas", df["Class"].nunique())

    st.divider()

    # Distribusi Class
    st.subheader("Distribusi Kelas Siswa")

    fig1 = px.histogram(
        df,
        x="Class",
        color="Class",
        text_auto=True
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Gender
    st.subheader("Distribusi Gender")

    fig2 = px.pie(
        df,
        names="gender",
        hole=0.4
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Topic
    st.subheader("Distribusi Topic")

    topic_count = df["Topic"].value_counts().reset_index()
    topic_count.columns = ["Topic", "Jumlah"]

    fig3 = px.bar(
        topic_count,
        x="Topic",
        y="Jumlah"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Korelasi
    st.subheader("Aktivitas Siswa")

    numeric_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion"
    ]

    fig4 = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color="Class"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # Filter
    st.subheader("Filter Data")

    kelas = st.selectbox(
        "Pilih Class",
        ["Semua"] + sorted(df["Class"].unique().tolist())
    )

    if kelas != "Semua":
        filtered_df = df[df["Class"] == kelas]
    else:
        filtered_df = df

    st.dataframe(filtered_df)

    # Statistik
    st.subheader("Statistik Deskriptif")

    st.dataframe(
        filtered_df[numeric_cols].describe()
    )

else:
    st.info("Silakan upload file CSV.")