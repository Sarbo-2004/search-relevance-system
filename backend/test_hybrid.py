import pickle
from retrieval.sbert import SBERTEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25 import BM25Retriever  # your existing BM25 class

# Load data
with open("data/df_nlp.pkl", "rb") as f:
    df = pickle.load(f)

# Load SBERT embeddings
encoder = SBERTEncoder(embedding_path="data/sbert_product_embeddings.pt")
doc_embeddings = encoder.get_precomputed_embeddings()

# Build FAISS
faiss_index = FaissIndex(doc_embeddings, df)

# Build BM25 (existing code)
bm25 = BM25Retriever(df, text_column="search_text")

# Hybrid retriever
hybrid = HybridRetriever(bm25, faiss_index)

# Query
query = "wireless noise cancelling headphones"
query_embedding = encoder.encode_query(query)

results = hybrid.search(query, query_embedding, top_k=5)

print(results[["title", "retrieval_source"]])
