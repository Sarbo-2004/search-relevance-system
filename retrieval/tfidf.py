import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFRetriever:
    def __init__(self, df: pd.DataFrame, text_column: str = "search_text"):
        self.df = df
        self.text_column = text_column
        self.corpus = df[text_column].tolist()
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, preprocess_fn=None, top_k: int = 5):
        # Preprocess query if a function is provided
        if preprocess_fn:
            query = preprocess_fn(query)

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        return self.df.iloc[top_k_indices]
