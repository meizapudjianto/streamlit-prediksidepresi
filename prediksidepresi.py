import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Memuat fungsi data
@st.cache_data
def load_data():
    data = pd.read_csv("responden_bersih.csv")  # Ensure this is your cleaned data file
    return data

# Fungsi untuk mengkategorikan tingkat depresi
def kategori_depresi(skor):
    if skor <= 9:
        return 'Normal'
    elif skor <= 13:
        return 'Ringan'
    elif skor <= 20:
        return 'Sedang'
    elif skor <= 27:
        return 'Parah'
    else:
        return 'Sangat Parah'

# Melatih dan mengevaluasi fungsi model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluation = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return y_pred, evaluation

# Muat data
data = load_data()
data['kategori_depresi'] = data['skor_depresi'].apply(kategori_depresi)

# Menyiapkan data pelatihan
X = data[['skor_stres', 'skor_kecemasan']]
y = data['skor_depresi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Menu Sidebar
st.sidebar.title("Prediksi Tingkat Depresi")
menu = st.sidebar.selectbox("Pilih Menu", ["Multiple Linear Regression", "Random Forest Regressor", "Gradient Boosting Regression"])

# Pemilihan model
if menu == "Multiple Linear Regression":
    st.title("Multiple Linear Regression")
    model = LinearRegression()
elif menu == "Random Forest Regressor":
    st.title("Random Forest Regressor")
    model = RandomForestRegressor(n_estimators=100, random_state=0)
elif menu == "Gradient Boosting Regression":
    st.title("Gradient Boosting Regression")
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)

# Melatih dan mengevaluasi model
y_pred, evaluation = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
y_pred_categories = pd.Series([kategori_depresi(score) for score in y_pred], index=y_test.index)

# Menampilkan metrik evaluasi
st.subheader("Hasil Evaluasi")
st.write("Mean Absolute Error (MAE):", evaluation["MAE"])
st.write("Root Mean Squared Error (RMSE):", evaluation["RMSE"])

# Menentukan urutan kategori yang diinginkan
category_order = ["Normal", "Ringan", "Sedang", "Parah", "Sangat Parah"]

# Plot distribusi data aktual vs prediksi
st.subheader("Distribusi Kategori Depresi")
col1, col2 = st.columns(2)

with col1:
    actual_counts = data['kategori_depresi'].value_counts(normalize=True).reindex(category_order).fillna(0).reset_index()
    actual_counts.columns = ['kategori_depresi', 'persentase']
    actual_counts['persentase'] *= 100
    bar_chart_actual = px.bar(
        actual_counts,
        x='kategori_depresi',
        y='persentase',
        labels={'persentase': 'Persentase (%)'},
        title="Data Aktual"
    )
    st.plotly_chart(bar_chart_actual)

with col2:
    pred_counts = y_pred_categories.value_counts(normalize=True).reindex(category_order).fillna(0).reset_index()
    pred_counts.columns = ['kategori_depresi', 'persentase']
    pred_counts['persentase'] *= 100
    bar_chart_pred = px.bar(
        pred_counts,
        x='kategori_depresi',
        y='persentase',
        labels={'persentase': 'Persentase (%)'},
        title=f"Prediksi"
    )
    st.plotly_chart(bar_chart_pred)

# Mempersiapkan data untuk perbandingan aktual vs prediksi berdasarkan demografi
# Data aktual yang dikelompokkan berdasarkan informasi demografis
data['kategori_depresi'] = pd.Categorical(data['kategori_depresi'], categories=category_order, ordered=True)

# Untuk setiap faktor demografis: angkatan, jenis_kelamin, tinggal_dengan_keluarga
for factor in ["angkatan", "jenis_kelamin", "tinggal_dengan_keluarga"]:
    st.subheader(f"Distribusi Kategori Depresi Berdasarkan {factor.capitalize()}")
    actual_grouped = data.groupby([factor, 'kategori_depresi']).size().reset_index(name='jumlah')
    actual_total = actual_grouped.groupby([factor])['jumlah'].transform('sum')
    actual_grouped['persentase'] = (actual_grouped['jumlah'] / actual_total) * 100

    bar_chart_actual = px.bar(
        actual_grouped,
        x=factor,
        y='persentase',
        color='kategori_depresi',
        category_orders={'kategori_depresi': category_order},
        labels={'persentase': 'Persentase (%)'},
        title=f"Data Aktual"
    )
    st.plotly_chart(bar_chart_actual)

    # Prediksi data yang dikelompokkan berdasarkan informasi demografis
    predicted_data = X_test.copy()
    predicted_data['kategori_prediksi'] = y_pred_categories
    predicted_data[factor] = data[factor].iloc[y_test.index].values

    predicted_grouped = predicted_data.groupby([factor, 'kategori_prediksi']).size().reset_index(name='jumlah')
    predicted_total = predicted_grouped.groupby([factor])['jumlah'].transform('sum')
    predicted_grouped['persentase'] = (predicted_grouped['jumlah'] / predicted_total) * 100
    predicted_grouped.rename(columns={'kategori_prediksi': 'kategori_depresi'}, inplace=True)

    bar_chart_pred = px.bar(
        predicted_grouped,
        x=factor,
        y='persentase',
        color='kategori_depresi',
        category_orders={'kategori_depresi': category_order},
        labels={'persentase': 'Persentase (%)'},
        title=f"Prediksi"
    )
    st.plotly_chart(bar_chart_pred)
