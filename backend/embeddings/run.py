from generate_sbert_embeddings import generate_sbert_embeddings
import pandas as pd

df_nlp = pd.read_pickle("data/df_nlp.pkl")
generate_sbert_embeddings(df_nlp, "search_text", "data/sbert_product_embeddings.pt")
