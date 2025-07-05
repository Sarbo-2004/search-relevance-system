from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np

class BM25Retriever:
    def __init__(self, df: pd.DataFrame, text_column: str = "search_text"):
        self.df = df
        self.text_column = text_column
        self.tokenized_corpus = [doc.split() for doc in df[text_column]]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, preprocess_fn=None, top_k: int = 5):
        if preprocess_fn:
            query = preprocess_fn(query)

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        return self.df.iloc[top_k_indices]
