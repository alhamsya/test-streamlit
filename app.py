import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("RandomForestClassifier_model.sav")

st.title("Prediksi Penyakit Jantung")
st.markdown("Masukkan data pasien untuk memprediksi kemungkinan risiko penyakit jantung.")

# Input user
age = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
chest_pain = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4])
resting_bp = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=50, max_value=200, value=130)
cholesterol = st.number_input("Kolesterol (mg/dL)", min_value=100, max_value=600, value=250)
fasting_blood_sugar = st.selectbox("Gula Darah Puasa > 120 mg/dL?", ["Tidak", "Ya"])
rest_ecg = st.selectbox("Hasil ECG Saat Istirahat", [0, 1, 2])
max_hr = st.slider("Detak Jantung Maks (bpm)", 70, 220, 150)
exercise_angina = st.selectbox("Angina Saat Olahraga?", ["Tidak", "Ya"])
oldpeak = st.number_input("Oldpeak (depresi ST)", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("Slope Segmen ST", [0, 1, 2])

# Konversi ke array input model
input_data = np.array([
    age,
    1 if sex == "Laki-laki" else 0,
    chest_pain,
    resting_bp,
    cholesterol,
    1 if fasting_blood_sugar == "Ya" else 0,
    rest_ecg,
    max_hr,
    1 if exercise_angina == "Ya" else 0,
    oldpeak,
    st_slope
]).reshape(1, -1)

# Prediksi dengan threshold manual (gunakan predict_proba)
if st.button("Prediksi"):
    proba = model.predict_proba(input_data)[0][1]  # Probabilitas penyakit (kelas 1)
    st.write(f"Probabilitas penyakit jantung: **{proba:.2f}**")

    if proba > 0.5:
        st.error("⚠️ Potensi Penyakit Jantung Terdeteksi.")
    else:
        st.success("✅ Tidak Terindikasi Penyakit Jantung.")