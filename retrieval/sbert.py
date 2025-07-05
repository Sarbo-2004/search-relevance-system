import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

class SBERTRetriever:
    def __init__(self, df, embedding_path, text_column="search_text"):
        self.df = df
        self.text_column = text_column
        self.texts = df[text_column].tolist()
        self.embeddings = torch.load(embedding_path, map_location=torch.device('cpu'))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, top_k=5):
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarity
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top_k results
        top_results = torch.topk(scores, k=top_k)
        top_indices = top_results.indices.cpu().numpy()
        
        return self.df.iloc[top_indices]
