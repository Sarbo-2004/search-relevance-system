import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def generate_sbert_embeddings(
    df: pd.DataFrame,
    text_column: str = "search_text",
    save_path: str = "data/sbert_product_embeddings.pt",
    model_name: str = "all-MiniLM-L6-v2",
    force_recompute: bool = False,
):
    
    if not force_recompute and os.path.exists(save_path):
        print(f"🔁 Embeddings already exist at {save_path}. Skipping generation.")
        return torch.load(save_path)

    print(f"📦 Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🧠 Encoding {len(df)} items from column: {text_column}")
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"✅ Saved embeddings to {save_path}")

    return embeddings
