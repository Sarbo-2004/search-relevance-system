import faiss
import numpy as np
import torch

class FaissIndex:
    def __init__(self, embeddings,df):
        self.df=df
        self.embeddings = embeddings.astype("float32")
        faiss.normalize_L2(self.embeddings)
        self.dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)
        
    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        results_df = self.df.iloc[indices[0]].reset_index(drop=True)
        results_df["score"] = scores[0]
        return results_df