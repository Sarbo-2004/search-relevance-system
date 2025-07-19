from fastapi import FastAPI, Query
from typing import List, Union
import pandas as pd
from retrieval.sbert import SBERTRetriever
from retrieval.bm25 import BM25Retriever 
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Search API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Load DataFrame
df = pd.read_pickle("data/df_nlp.pkl")

# Load Search Engines
sbert_engine = SBERTRetriever(df, "data/sbert_product_embeddings.pt")
bm25_engine = BM25Retriever(df)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search(query: str = Query(...), top_k: int = 5):
    try:
        results, scores = sbert_engine.search(query, top_k, return_scores=True)
        threshold = 0.4  # Set a similarity threshold for SBERT
    except Exception as e:
        print("SBERT failed, falling back to BM25:", e)
        results, scores = bm25_engine.search(query, top_k, return_scores=True)
        threshold = 0.2  # Lower threshold for BM25

    # Filter results by threshold
    filtered = []
    for i, score in enumerate(scores):
        if score >= threshold:
            row = results.iloc[i]
            filtered.append({
                "title": row["title"],
                "price": row["price"],
                "search_text": row["search_text"],
                "images": row["images"],
                "categories": row["categories"],
                "score": round(float(score), 4)
            })

    if not filtered:
        return {"message": "No relevant results found for your query."}
    
    return filtered

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
