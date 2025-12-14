import pickle
from retrieval.sbert import SBERTEncoder
from retrieval.faiss_index import FaissIndex

with open("data/df_nlp.pkl","rb") as f:
    df = pickle.load(f)

encoder = SBERTEncoder(embedding_path="data/sbert_product_embeddings.pt")
doc_embeddings = encoder.get_precomputed_embeddings()

faiss_index = FaissIndex(doc_embeddings, df)
query = "wireless noise cancelling headphones"
query_embedding = encoder.encode_query(query)
results = faiss_index.search(query_embedding, top_k=5)

print(results[["title", "score"]])