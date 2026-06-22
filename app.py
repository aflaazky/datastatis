import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =====================================
# KONFIGURASI HALAMAN
# =====================================
st.set_page_config(
    page_title="Dashboard Analisis Data",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Data")
st.markdown("Aplikasi untuk eksplorasi, visualisasi, dan prediksi data.")

# =====================================
# UPLOAD DATASET
# =====================================
uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

# =====================================
# MEMBACA DATA
# =====================================
df = pd.read_csv(uploaded_file)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("Informasi Dataset")

st.sidebar.write(f"Baris : {df.shape[0]}")
st.sidebar.write(f"Kolom : {df.shape[1]}")

# =====================================
# OVERVIEW DATA
# =====================================
st.header("📋 Overview Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Jumlah Baris", df.shape[0])

with col2:
    st.metric("Jumlah Kolom", df.shape[1])

st.dataframe(df.head())

# =====================================
# INFORMASI KOLOM
# =====================================
st.header("📌 Tipe Data")

dtype_df = pd.DataFrame({
    "Kolom": df.columns,
    "Tipe Data": df.dtypes.astype(str)
})

st.dataframe(dtype_df)

# =====================================
# MISSING VALUE
# =====================================
st.header("🔍 Missing Value")

missing_df = pd.DataFrame({
    "Kolom": df.columns,
    "Jumlah Missing": df.isnull().sum()
})

st.dataframe(missing_df)

# =====================================
# STATISTIK DESKRIPTIF
# =====================================
st.header("📈 Statistik Deskriptif")

st.dataframe(df.describe())

# =====================================
# VISUALISASI NUMERIK
# =====================================
numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

if len(numeric_cols) > 0:

    st.header("📊 Visualisasi Data")

    selected_col = st.selectbox(
        "Pilih Kolom Numerik",
        numeric_cols
    )

    fig = px.histogram(
        df,
        x=selected_col,
        nbins=20,
        title=f"Distribusi {selected_col}"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# KORELASI
# =====================================
if len(numeric_cols) > 1:

    st.header("🔥 Heatmap Korelasi")

    corr = df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

# =====================================
# MACHINE LEARNING
# =====================================
st.header("🤖 Prediksi Data")

target_column = st.selectbox(
    "Pilih Kolom Target",
    df.columns
)

if st.button("Jalankan Model"):

    data_ml = df.copy()

    for col in data_ml.columns:
        if data_ml[col].dtype == "object":
            le = LabelEncoder()
            data_ml[col] = le.fit_transform(
                data_ml[col].astype(str)
            )

    X = data_ml.drop(columns=[target_column])
    y = data_ml[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.success(
        f"Akurasi Model : {accuracy:.2%}"
    )

    importance = pd.DataFrame({
        "Fitur": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig_imp = px.bar(
        importance,
        x="Importance",
        y="Fitur",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )
