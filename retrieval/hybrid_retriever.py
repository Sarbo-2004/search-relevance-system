import pandas as pd

class HybridRetriever:
    
    def __init__(self, bm25_retriever, faiss_index):
        self.bm25 = bm25_retriever
        self.faiss = faiss_index
    
    def search(self, query, query_embedding,top_k=5):
        bm25_results = self.bm25.search(query, top_k=top_k)
        bm25_results["retrieval_source"] = "bm25"
        
        faiss_results = self.faiss.search(query_embedding, top_k=top_k)
        faiss_results["retrieval_source"] = "faiss"
        
        combined = pd.concat([bm25_results, faiss_results], ignore_index=True)
        combined = combined.drop_duplicates(subset=["title"], keep="first")
        
        return combined.reset_index(drop=True)