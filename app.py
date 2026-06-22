import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ====================================
# KONFIGURASI HALAMAN
# ====================================
st.set_page_config(
    page_title="Dashboard Edu Data",
    layout="wide"
)

st.title("📚 Dashboard Analisis Data Pendidikan")

# ====================================
# UPLOAD FILE
# ====================================
uploaded_file = st.file_uploader(
    "Upload file xAPI-Edu-Data.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload dataset terlebih dahulu.")
    st.stop()

# ====================================
# LOAD DATA
# ====================================
df = pd.read_csv(uploaded_file)

# ====================================
# DATASET
# ====================================
st.header("Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Jumlah Baris", df.shape[0])

with col2:
    st.metric("Jumlah Kolom", df.shape[1])

st.dataframe(df.head())

# ====================================
# INFORMASI DATA
# ====================================
st.header("Informasi Dataset")

st.write(df.dtypes)

# ====================================
# MISSING VALUE
# ====================================
st.header("Missing Value")

missing = df.isnull().sum()

st.dataframe(missing)

# ====================================
# DISTRIBUSI CLASS
# ====================================
st.header("Distribusi Class")

if "Class" in df.columns:

    class_count = df["Class"].value_counts().reset_index()

    class_count.columns = ["Class", "Jumlah"]

    fig = px.bar(
        class_count,
        x="Class",
        y="Jumlah",
        title="Distribusi Kelas Siswa",
        text="Jumlah"
    )

    st.plotly_chart(fig, use_container_width=True)

# ====================================
# ANALISIS FITUR NUMERIK
# ====================================
st.header("Analisis Fitur Numerik")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(numeric_cols) > 0:

    fitur = st.selectbox(
        "Pilih Fitur",
        numeric_cols
    )

    fig2 = px.histogram(
        df,
        x=fitur,
        nbins=20,
        title=f"Distribusi {fitur}"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ====================================
# KORELASI
# ====================================
st.header("Heatmap Korelasi")

if len(numeric_cols) > 1:

    corr = df[numeric_cols].corr()

    fig3 = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ====================================
# MACHINE LEARNING
# ====================================
st.header("Prediksi Class Menggunakan Decision Tree")

if "Class" in df.columns:

    data_ml = df.copy()

    for col in data_ml.columns:
        if data_ml[col].dtype == "object":

            le = LabelEncoder()

            data_ml[col] = le.fit_transform(
                data_ml[col].astype(str)
            )

    X = data_ml.drop("Class", axis=1)

    y = data_ml["Class"]

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

    acc = accuracy_score(
        y_test,
        y_pred
    )

    st.success(
        f"Akurasi Model: {acc:.2%}"
    )

# ====================================
# FEATURE IMPORTANCE
# ====================================
st.header("Feature Importance")

importance = pd.DataFrame({
    "Fitur": X.columns,
    "Nilai": model.feature_importances_
})

importance = importance.sort_values(
    by="Nilai",
    ascending=False
)

fig4 = px.bar(
    importance,
    x="Nilai",
    y="Fitur",
    orientation="h"
)

st.plotly_chart(fig4, use_container_width=True)
