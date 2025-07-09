from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from retrieval.sbert import SBERTRetriever
from retrieval.bm25 import BM25Retriever 
import uvicorn
import os

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's dynamic port
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)