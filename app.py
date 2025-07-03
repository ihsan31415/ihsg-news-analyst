import streamlit as st
import matplotlib.pyplot as plt
from pipeline import NewsToPricePipeline

# Load the model
model = NewsToPricePipeline()

# Streamlit App Layout
st.set_page_config(page_title="IHSG Forecast from News", layout="centered")
st.title("📰📈 Prediksi Dampak Berita terhadap IHSG")

with st.form("prediction_form"):
    judul = st.text_input("Judul Berita", "IHSG turun dan anjlok sampai akhir pekan")
    isi = st.text_area("Isi Berita", """
IHSG turun anjlok sampai akhir pekan, penurunan ini disebabkan oleh beberapa faktor seperti ketidakpastian ekonomi global, penurunan harga komoditas, dan sentimen negatif dari investor.
Penurunan ini juga dipicu oleh aksi jual besar-besaran dari investor asing yang mengakibatkan IHSG turun tajam.
Meskipun demikian, beberapa analis masih optimis bahwa IHSG akan pulih dalam jangka pendek, terutama jika ada perbaikan dalam kondisi ekonomi global dan stabilitas politik di Indonesia.
""")
    tanggal = st.date_input("Tanggal Berita")
    submitted = st.form_submit_button("🔮 Prediksi")

if submitted:
    st.info("Menghitung prediksi untuk 7 horizon ke depan...")
    try:
        from datetime import datetime
        tanggal_str = tanggal.strftime("%Y-%m-%d")

        horizons = list(range(1, 8))
        predictions = []

        for h in horizons:
            pred = model(judul, isi, tanggal_str, h)
            predictions.append(pred)

        # Plotting
        st.subheader("📊 Grafik Prediksi")
        fig, ax = plt.subplots()
        ax.plot(horizons, predictions, marker="o", linestyle="-", color="blue")
        ax.set_title("Prediksi Perubahan Harga IHSG")
        ax.set_xlabel("Horizon (Hari ke Depan)")
        ax.set_ylabel("Prediksi Perubahan (%)")
        ax.grid(True)
        st.pyplot(fig)

        # Table
        st.subheader("📋 Tabel Hasil Prediksi")
        for h, p in zip(horizons, predictions):
            st.write(f"Horizon {h}: **{p:.4f}%**")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
