import pandas as pd
from retrieval.sbert import SBERTSemanticRetriever
from query_processing.normalize import preprocess_query


def main():
    # Load your preprocessed product data
    df_nlp = pd.read_pickle("data/df_nlp.pkl")

    # Path to cached SBERT embeddings
    embedding_path = "data/sbert_product_embeddings.pt"

    # Initialize SBERT retriever with precomputed embeddings
    retriever = SBERTSemanticRetriever(
        df=df_nlp,
        embedding_cache_path=embedding_path
    )

    # Run a test query
    query = input("🔍 Enter your search query: ")

    # Retrieve top results
    results = retriever.search(query, preprocess_fn=preprocess_query, top_k=5)

    # Display the results
    print("\n🎯 Top 5 Results:")
    print(results[["title", "price", "search_text"]].to_string(index=False))


if __name__ == "__main__":
    main()
