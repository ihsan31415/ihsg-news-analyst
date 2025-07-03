import numpy as np
import pandas as pd
import torch
import math
import os
import calendar
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import random
import torch.nn as nn

#loading my models
sbert_model = SentenceTransformer("ihsan31415/large-fine-tuned-ihsg-sentence-bert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ihsan31415/indo-roBERTa-financial-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("ihsan31415/indo-roBERTa-financial-sentiment")

# Load topic embedding
topic_embedding = torch.load("data/topic_embeddings.pt")
if topic_embedding.dim() == 2:
    topic_embedding = topic_embedding.squeeze(0)
topic_embedding_np = topic_embedding.unsqueeze(0).numpy()

# Load horizon embedding layer


def get_horizon_embedding_layer(seed=37, embedding_dim=4, save_path="data/horizon_embedding_model.pt"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    horizon_values = torch.tensor([1, 2, 3, 4, 5, 6, 7]) - 1
    horizon_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
    embedded_horizon = horizon_embedding(horizon_values)
    torch.save(horizon_embedding.state_dict(), save_path)
    print(embedded_horizon)
    return horizon_embedding

# Initialize horizon embedding layer
horizon_embedding = get_horizon_embedding_layer()

# Load holidays CSV (expects a column named 'date' in YYYY-MM-DD format)
holidays_df = pd.read_csv("data/holidays.csv")
holiday_dates = set(holidays_df['date'].astype(str))

def is_holiday(date_str):
    return date_str in holiday_dates

def cyclical_encode(value, max_value):
    sin_val = math.sin(2 * math.pi * value / max_value)
    cos_val = math.cos(2 * math.pi * value / max_value)
    return sin_val, cos_val

def compute_logits(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    return outputs.logits.squeeze().tolist()

def transform_input(judul, isi, date_str, horizon_int):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    target_date = date_obj + timedelta(days=horizon_int)

    # Temporal features
    day_of_month = target_date.day
    month = target_date.month - 1
    year = target_date.year
    day_of_year = target_date.timetuple().tm_yday
    total_days_in_month = (datetime(year, month + 2, 1) - timedelta(days=1)).day if month < 11 else 31
    total_days_in_year = 366 if calendar.isleap(year) else 365

    month_percentage = day_of_month / total_days_in_month
    year_percentage = day_of_year / total_days_in_year

    dow = target_date.weekday()  # 0 = Monday, 6 = Sunday
    is_weekend = int(dow >= 5)
    date_fmt = target_date.strftime("%Y-%m-%d")
    is_holiday_flag = int(is_holiday(date_fmt))
    is_trading_day = int(not is_weekend and not is_holiday_flag)

    year_quarter = (month // 3)
    month_quarter = (day_of_month - 1) // (total_days_in_month // 4)

    dow_sin, dow_cos = cyclical_encode(dow, 7)
    month_sin, month_cos = cyclical_encode(month, 12)
    yq_sin, yq_cos = cyclical_encode(year_quarter, 4)
    mq_sin, mq_cos = cyclical_encode(month_quarter, 4)

    # Horizon embedding lookup
    horizon_tensor = torch.tensor([horizon_int - 1])
    horizon_embed_tensor = horizon_embedding(horizon_tensor).squeeze(0)
    horizon_embed = horizon_embed_tensor.tolist()

    # Embedding-based features
    judul_emb = sbert_model.encode(judul, convert_to_numpy=True).reshape(1, -1)
    isi_emb = sbert_model.encode(isi, convert_to_numpy=True).reshape(1, -1)
    relv_judul = cosine_similarity(judul_emb, topic_embedding_np)[0][0]
    relv_isi = cosine_similarity(isi_emb, topic_embedding_np)[0][0]

    # Sentiment logits
    judul_logits = compute_logits(judul)
    isi_logits = compute_logits(isi)

    features = {
        'relv_judul': relv_judul,
        'relv_isi': relv_isi,
        'judul_logits_0': judul_logits[0],
        'judul_logits_1': judul_logits[1],
        'judul_logits_2': judul_logits[2],
        'isi_logits_0': isi_logits[0],
        'isi_logits_1': isi_logits[1],
        'isi_logits_2': isi_logits[2],
        'month_percentage': month_percentage,
        'year_percentage': year_percentage,
        'is_holiday': is_holiday_flag,
        'is_weekend': is_weekend,
        'is_trading_day': is_trading_day,
        'day_of_week_sin': dow_sin,
        'day_of_week_cos': dow_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'year_quarter_sin': yq_sin,
        'year_quarter_cos': yq_cos,
        'month_quarter_sin': mq_sin,
        'month_quarter_cos': mq_cos,
        'horizon_embed_0': horizon_embed[0],
        'horizon_embed_1': horizon_embed[1],
        'horizon_embed_2': horizon_embed[2],
        'horizon_embed_3': horizon_embed[3],
    }
    return features
