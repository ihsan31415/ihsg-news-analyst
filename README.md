# 📰 IHSG News-to-Price Forecasting Pipeline

This project forecasts IHSG (Indonesia Stock Exchange Composite Index) price changes using a custom-built MLP model powered by financial news relevance and sentiment. It utilizes fine-tuned SBERT and RoBERTa models to extract signal from Indonesian news articles and maps them to a numerical price change forecast.

---

## 📊 What It Does

Given a news **title**, **content**, and a **date**, the model:

1. Measures how relevant the news is to IHSG topics using fine-tuned SBERT.
2. Computes sentiment logits using a fine-tuned Indonesian financial RoBERTa model.
3. Extracts time-based features (e.g., holidays, quarter progression, cyclic encoding).
4. Encodes the forecast horizon (1–7 days) via `nn.Embedding`.
5. Predicts the percentage change in IHSG price for future trading days.

---

## 🏗️ Tech Stack

- PyTorch (MLP model)
- `transformers` & `sentence-transformers` (SBERT & RoBERTa)
- Scikit-learn (scaling, cosine similarity, joblib)
- Streamlit (web UI)
- Hugging Face Hub

---

## 📁 Project Structure

```

├── app.py                    # Streamlit interface
├── pipeline.py               # Main prediction pipeline
├── predict.py                # CLI-based prediction runner
├── requirements.txt          # Required packages

├── config/
│   └── config.json           # (Optional) Configurations and settings

├── data/
│   ├── holidays.csv
│   ├── horizon\_embedding\_model.pt
│   ├── scaler\_x.pkl
│   ├── scaler\_y.pkl
│   ├── topic\_embeddings.pt
│   └── topic\_phrases.csv

├── features/
│   ├── topic\_embedding.py    # Generates topic embedding from phrases
│   └── transform.py          # Feature engineering and input transformation

├── models/
│   ├── horizon\_embedding.py  # Generates horizon embedding
│   ├── load\_model.py         # Loads trained MLP model
│   └── main\_model.pth        # Trained model weights

````

---

## ⚙️ Installation & Setup

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/ihsg-news-analyst.git
cd ihsg-news-analyst
````

### 2. Create a Virtual Environment

<details>
<summary>📌 On Linux/macOS</summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

</details>

<details>
<summary>📌 On Windows</summary>

```bash
python -m venv venv
venv\Scripts\activate
```

</details>

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Initialize Topic & Horizon Embeddings

These scripts will generate and save required model weights:

```bash
python features/topic_embedding.py
python models/horizon_embedding.py
```

---

## 🚀 Usage

### 📦 Run Prediction Programmatically (Pipeline Example)

```python
from pipeline import NewsToPricePipeline

model = NewsToPricePipeline()

judul = "IHSG Merosot Tajam Jelang Akhir Pekan"
isi = """
Indeks Harga Saham Gabungan (IHSG) mengalami penurunan tajam menjelang akhir pekan, terdampak oleh sejumlah faktor eksternal dan domestik. Ketidakpastian ekonomi global, penurunan harga komoditas, serta sentimen negatif dari para investor menjadi penyebab utama pelemahan indeks.

Aksi jual besar-besaran oleh investor asing juga turut memperburuk tekanan terhadap pasar, menyebabkan IHSG anjlok lebih dalam. Meskipun demikian, sebagian analis masih memandang positif prospek jangka pendek IHSG, dengan catatan adanya perbaikan dalam kondisi makroekonomi global dan stabilitas politik nasional.

Investor diimbau untuk tetap berhati-hati dan melakukan diversifikasi portofolio guna mengurangi risiko kerugian akibat volatilitas pasar yang tinggi.
"""
tanggal = "2025-09-02"  # YYYY-MM-DD format
horizon = 4

pred = model(judul, isi, tanggal, horizon)
print(f"Predicted price change: {pred:.4f}")
```

---

### 🖥️ Run Streamlit App

```bash
streamlit run app.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📬 Author & Credits

* 👤 **@ihsan31415** (Model development, fine-tuning, data pipeline)
* 💼 Powered by custom fine-tuned Indonesian NLP models on top of open-source foundations.

---

## 📌 Notes

* SBERT and RoBERTa models are fine-tuned specifically for Indonesian financial domain tasks.
* Horizon-aware prediction using learned embeddings enables multi-step forecasts.

---

## 📜 License

This project is for research and demonstration purposes. Contact the author for potential production or commercial usage.
