import numpy
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("ihsan31415/large-fine-tuned-ihsg-sentence-bert")

df_topics = pd.read_csv('data/topic_phrases.csv')
topic_phrases_from_csv = df_topics['Topic Phrase'].tolist()

topic_embeddings_csv = sbert_model.encode(topic_phrases_from_csv, convert_to_tensor=True)
embedding_topic = torch.mean(topic_embeddings_csv, dim=0)
torch.save(embedding_topic, 'data/topic_embeddings.pt')

print("Topic phrase embeddings computed and saved to data/topic_embeddings.pt")