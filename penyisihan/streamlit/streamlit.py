import pandas as pd
import xgboost as xgb
import streamlit as st
import joblib
import matplotlib.pyplot as plt


model = xgb.Booster(model_file="/mount/src/2024-q4-fortex-ai/penyisihan/model/model.json")
X = joblib.load("/mount/src/2024-q4-fortex-ai/penyisihan/model/df.pkl")

# UI Streamlit
st.set_page_config(
    page_title="Prediksi Konsumsi Listrik Bulanan", page_icon="🏢", layout="wide"
)

st.title("Prediksi Konsumsi Listrik Bulanan ⚡")

# Deskripsi
st.write(
    "Isi formulir berikut untuk memberikan input pada model prediksi konsumsi listrik bulanan."
    " Data yang Anda masukkan akan digunakan untuk menghasilkan prediksi yang akurat."
)

# Membuat input form dengan beberapa kategori
with st.form(key="electricity_form"):
    st.header("Informasi Bangunan")

    building_type = st.selectbox("Tipe Bangunan", X["BuildingType"].unique().tolist())

    peak_usage_hour = st.number_input(
        "Jam Penggunaan Puncak (0-23)", min_value=0, max_value=23, value=12
    )

    st.header("Kapasitas Energi Terbarukan")

    renewable_capacity = st.number_input(
        "Kapasitas Energi Terbarukan (kWh)", min_value=0.0, value=25000.0, step=100.0
    )
    renewable_type = st.selectbox(
        "Tipe Energi Terbarukan",
        [x for x in X["RenewableType"].unique().tolist() if pd.notna(x)],
    )
    energy_source = st.selectbox("Sumber Energi", X["EnergySource"].unique().tolist())

    st.header("Efisiensi Energi & Data Cuaca")

    energy_efficiency = st.number_input(
        "Efisiensi Energi (kWh/m2)", min_value=5.0, max_value=50.0, value=25.0, step=1.0
    )
    temperature = st.number_input(
        "Suhu (°C)", min_value=-20.0, max_value=45.0, value=20.0, step=0.5
    )
    solar_intensity = st.number_input(
        "Intensitas Matahari (Jam)", min_value=0.0, max_value=12.0, value=6.0, step=0.1
    )
    wind_speed = st.number_input(
        "Kecepatan Angin (km/h)", min_value=0.0, max_value=120.0, value=30.0, step=1.0
    )

    submit_button = st.form_submit_button("Prediksi Konsumsi Listrik")


# Fungsi prediksi
def make_prediction(inputs):
    # Membuat dataframe untuk inputan pengguna
    df = pd.DataFrame(
        [inputs],
        columns=[
            "BuildingType",
            "PeakUsageTime_Hour",
            "RenewableCapacity_kWh",
            "RenewableType",
            "EnergySource",
            "EnergyEfficiency_kWh_per_m2",
            "WeatherData_Temperature_C",
            "WeatherData_SolarIntensity_Hours",
            "WeatherData_WindSpeed_km_h",
        ],
    )

    # Encoding untuk kolom kategorikal
    df["BuildingType"] = (
        df["BuildingType"]
        .astype("category")
        .cat.set_categories(X["BuildingType"].cat.categories)
    )
    df["RenewableType"] = (
        df["RenewableType"]
        .astype("category")
        .cat.set_categories(X["RenewableType"].cat.categories)
    )
    df["EnergySource"] = (
        df["EnergySource"]
        .astype("category")
        .cat.set_categories(X["EnergySource"].cat.categories)
    )

    # Menggunakan model untuk regresi
    prediction = model.predict(xgb.DMatrix(df, enable_categorical=True))
    return float(prediction[0])  # Mengembalikan nilai prediksi


if submit_button:
    # Mengumpulkan data input dari form
    inputs = {
        "BuildingType": building_type,
        "PeakUsageTime_Hour": peak_usage_hour,
        "RenewableCapacity_kWh": renewable_capacity,
        "RenewableType": renewable_type,
        "EnergySource": energy_source,
        "EnergyEfficiency_kWh_per_m2": energy_efficiency,
        "WeatherData_Temperature_C": temperature,
        "WeatherData_SolarIntensity_Hours": solar_intensity,
        "WeatherData_WindSpeed_km_h": wind_speed,
    }

    prediction = make_prediction(inputs)

    # Membandingkan prediksi untuk setiap tipe energi terbarukan
    renewable_types = ["Tidal", "Solar", "Wind", "Geothermal", "Hydro", "Biomass"]
    predictions = []

    for r_type in renewable_types:
        inputs["RenewableType"] = r_type
        predictions.append(make_prediction(inputs))

    # Mengurutkan hasil prediksi berdasarkan nilai
    sorted_results = sorted(
        zip(renewable_types, predictions), key=lambda x: x[1], reverse=True
    )
    sorted_types, sorted_predictions = zip(*sorted_results)

    # Visualisasi hasil prediksi untuk semua tipe energi terbarukan
    st.subheader("Perbandingan Prediksi untuk Setiap Tipe Energi Terbarukan")
    fig, ax = plt.subplots()
    ax.bar(sorted_types, sorted_predictions, color="#3498db")
    ax.set_ylabel("Prediksi Konsumsi Listrik Bulanan (kWh)")
    ax.set_title("Hasil Prediksi Berdasarkan Tipe Energi Terbarukan")
    st.pyplot(fig)

    # Menampilkan hasil prediksi
    st.markdown(
        f"**Prediksi Konsumsi Listrik Bulanan:** {prediction:.2f} kWh untuk {renewable_type}"
    )
