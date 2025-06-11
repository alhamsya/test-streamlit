import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("RandomForestClassifier_model.sav")

# Title
st.title("Prediksi Risiko Kematian Pasien Gagal Jantung")
st.markdown("Masukkan data pasien untuk memprediksi kemungkinan risiko kematian berdasarkan model klasifikasi Random Forest.")

# Input fields (11 fitur)
age = st.number_input("Umur (tahun)", min_value=0, max_value=120, value=60)
anaemia = st.selectbox("Anemia", ["Tidak", "Ya"])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=200)
diabetes = st.selectbox("Diabetes", ["Tidak", "Ya"])
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 35)
high_blood_pressure = st.selectbox("Hipertensi", ["Tidak", "Ya"])
platelets = st.number_input("Platelet (kiloplatelets/mL)", min_value=0, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0)
serum_sodium = st.slider("Serum Sodium (mEq/L)", 100, 150, 137)
sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
smoking = st.selectbox("Merokok", ["Tidak", "Ya"])

# NOTE: Jangan sertakan "time" jika model tidak dilatih dengannya!

# Convert input to model-compatible array
input_data = np.array([
    age,
    1 if anaemia == "Ya" else 0,
    creatinine_phosphokinase,
    1 if diabetes == "Ya" else 0,
    ejection_fraction,
    1 if high_blood_pressure == "Ya" else 0,
    platelets,
    serum_creatinine,
    serum_sodium,
    1 if sex == "Laki-laki" else 0,
    1 if smoking == "Ya" else 0
]).reshape(1, -1)

# Prediction
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ Pasien memiliki risiko tinggi kematian.")
    else:
        st.success("✅ Pasien memiliki risiko rendah kematian.")