import pickle
from retrieval.sbert import SBERTEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25 import BM25Retriever
from rag.rag_chain import build_rag_chain

with open("data/df_nlp.pkl", "rb") as f:
    df = pickle.load(f)

encoder = SBERTEncoder("data/sbert_product_embeddings.pt")
faiss_index = FaissIndex(encoder.get_precomputed_embeddings(), df)
bm25 = BM25Retriever(df, text_column="search_text")

hybrid = HybridRetriever(bm25, faiss_index)
rag_chain = build_rag_chain(hybrid, encoder)

query = "Which noise cancelling headphones are good for travel?"
response = rag_chain.invoke(query)

print(response)
