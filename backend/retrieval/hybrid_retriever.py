import pandas as pd


def min_max_normalize(series):
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())


class HybridRetriever:
    def __init__(self, bm25_retriever, faiss_index, alpha=0.4, beta=0.6):
        self.bm25 = bm25_retriever
        self.faiss = faiss_index
        self.alpha = alpha
        self.beta = beta

    def search(self, query, query_embedding, top_k=5):
        # 1️⃣ Retrieve candidates
        bm25_df = self.bm25.search(query, top_k=top_k * 3)
        bm25_df["bm25_score"] = bm25_df["score"]
        bm25_df["faiss_score"] = 0.0
        bm25_df["retrieval_source"] = "bm25"

        faiss_df = self.faiss.search(query_embedding, top_k=top_k * 3)
        faiss_df["faiss_score"] = faiss_df["score"]
        faiss_df["bm25_score"] = 0.0
        faiss_df["retrieval_source"] = "faiss"

        # 2️⃣ Combine
        combined = pd.concat([bm25_df, faiss_df], ignore_index=True)

        # 3️⃣ Deduplicate
        combined = combined.drop_duplicates(subset=["title"], keep="first")

        # 4️⃣ Normalize
        combined["bm25_norm"] = min_max_normalize(combined["bm25_score"])
        combined["faiss_norm"] = min_max_normalize(combined["faiss_score"])

        # 5️⃣ Score fusion
        combined["final_score"] = (
            self.alpha * combined["bm25_norm"] +
            self.beta * combined["faiss_norm"]
        )

        # 6️⃣ Global ranking
        combined = combined.sort_values(
            by="final_score", ascending=False
        )

        return combined.head(top_k).reset_index(drop=True)
