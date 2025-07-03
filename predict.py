from pipeline import NewsToPricePipeline

model = NewsToPricePipeline()

judul = "ihsg turun dan anjlok sampai akhir pekan"
isi = """
ihsg turun anjlok sampai akhir pekan, penurunan ini disebabkan oleh beberapa faktor seperti ketidakpastian ekonomi global, penurunan harga komoditas, dan sentimen negatif dari investor.
Penurunan ini juga dipicu oleh aksi jual besar-besaran dari investor asing yang mengakibatkan IHSG turun tajam.
Meskipun demikian, beberapa analis masih optimis bahwa IHSG akan pulih dalam jangka pendek, terutama jika ada perbaikan dalam kondisi ekonomi global dan stabilitas politik di Indonesia.
Investor disarankan untuk tetap waspada dan melakukan diversifikasi portofolio untuk mengurangi risiko kerugian.
IHSG turun tajam pada akhir pekan ini, penurunan ini disebabkan oleh beberapa faktor seperti ketidakpastian ekonomi global, penurunan harga komoditas, dan sentimen negatif dari investor.
Penurunan ini juga dipicu oleh aksi jual besar-besaran dari investor asing yang mengakibatkan IHSG turun tajam.
Meskipun demikian, beberapa analis masih optimis bahwa IHSG akan pulih dalam jangka pendek, terutama jika ada perbaikan dalam kondisi ekonomi global dan stabilitas politik di Indonesia.
"""
tanggal = "2025-09-02"  # YYYY-MM-DD format
horizon = 4


pred = model(judul, isi, tanggal, horizon)
print(f"Predicted price change: {pred:.4f}")