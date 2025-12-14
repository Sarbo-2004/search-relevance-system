import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, df, text_column="search_text"):
        self.df = df.reset_index(drop=True)
        self.text_column = text_column

        corpus = self.df[text_column].astype(str).tolist()
        tokenized_corpus = [doc.lower().split() for doc in corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        result_df = self.df.iloc[top_indices].copy()
        result_df["score"] = top_scores

        return result_df.reset_index(drop=True)
