import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======================
# LOAD DATA
# ======================
st.set_page_config(page_title="Dashboard Edu Data", layout="wide")

st.title("📊 Dashboard Analisis xAPI Edu Data")

df = pd.read_csv("xAPI-Edu-Data.csv")

# ======================
# DATASET
# ======================
st.header("Dataset")

st.write("Jumlah Data :", df.shape[0])
st.write("Jumlah Kolom :", df.shape[1])

st.dataframe(df.head())

# ======================
# STATISTIK
# ======================
st.header("Statistik Deskriptif")

st.dataframe(df.describe())

# ======================
# DISTRIBUSI KELAS
# ======================
st.header("Distribusi Class")

class_count = df["Class"].value_counts().reset_index()
class_count.columns = ["Class", "Jumlah"]

fig = px.bar(
    class_count,
    x="Class",
    y="Jumlah",
    color="Class",
    title="Distribusi Kelas Siswa"
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# FITUR NUMERIK
# ======================
st.header("Analisis Fitur Numerik")

numeric_cols = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion"
]

selected_feature = st.selectbox(
    "Pilih Fitur",
    numeric_cols
)

fig2 = px.histogram(
    df,
    x=selected_feature,
    color="Class",
    nbins=20,
    title=f"Distribusi {selected_feature}"
)

st.plotly_chart(fig2, use_container_width=True)

# ======================
# KORELASI
# ======================
st.header("Heatmap Korelasi")

df_corr = df.copy()

le = LabelEncoder()

for col in df_corr.columns:
    if df_corr[col].dtype == "object":
        df_corr[col] = le.fit_transform(df_corr[col])

corr = df_corr.corr()

fig3 = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Matriks Korelasi"
)

st.plotly_chart(fig3, use_container_width=True)

# ======================
# MACHINE LEARNING
# ======================
st.header("Prediksi Class Siswa")

data_ml = df.copy()

encoders = {}

for col in data_ml.columns:
    if data_ml[col].dtype == "object":
        le = LabelEncoder()
        data_ml[col] = le.fit_transform(data_ml[col])
        encoders[col] = le

X = data_ml.drop("Class", axis=1)
y = data_ml["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc:.2%}")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.subheader("Classification Report")

st.dataframe(pd.DataFrame(report).transpose())

# ======================
# FEATURE IMPORTANCE
# ======================
st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig4 = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Pengaruh Fitur terhadap Class"
)

st.plotly_chart(fig4, use_container_width=True)
