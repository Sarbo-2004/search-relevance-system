from retrieval.sbert import SBERTRetriever
from retrieval.bm25 import BM25Retriever
import pandas as pd

class SearchEngine:
    def __init__(self):
        self.df = pd.read_pickle("data/df_nlp.pkl")
        self.sbert = SBERTRetriever(self.df, "data\sbert_product_embeddings.pt")
        self.bm25 = BM25Retriever(self.df)

    def search(self, query: str, top_k: int = 5):
        try:
            results = self.sbert.search(query, top_k)
        except:
            results = self.bm25.search(query, top_k)
        return results[["title", "price", "search_text"]].to_dict(orient="records")
