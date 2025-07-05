from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from retrieval.sbert import SBERTRetriever
from retrieval.bm25 import BM25Retriever  # Optional fallback

app = FastAPI(title="Smart Search API")

# Load DataFrame
df = pd.read_pickle("data/df_nlp.pkl")

# Load Search Engines
sbert_engine = SBERTRetriever(df, "data/sbert_product_embeddings.pt")
bm25_engine = BM25Retriever(df)  # Optional fallback

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search(query: str = Query(...), top_k: int = 5):
    try:
        results = sbert_engine.search(query, top_k)
    except Exception as e:
        print("SBERT failed, falling back to BM25:", e)
        results = bm25_engine.search(query, top_k)

    return results[["title", "price", "search_text"]].to_dict(orient="records")
