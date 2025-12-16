import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    return pd.read_csv("data/restaurants.csv")

def prepare_embeddings(df):
    texts = (
        df["cuisine"] + " " +
        df["price_range"] + " " +
        df["description"]
    ).tolist()
    return model.encode(texts)

def recommend(user_query, df, embeddings, top_k=3):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    df["score"] = similarities
    top_results = df.sort_values("score", ascending=False).head(top_k)

    return top_results
