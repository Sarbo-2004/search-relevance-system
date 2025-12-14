import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class SBERTEncoder:
    """
    SBERT Encoder class
    Responsibility:
    - Load SBERT model
    - Generate embeddings (documents / query)
    """

    def __init__(self, embedding_path=None, text_column="search_text"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Optional: load precomputed embeddings
        self.embeddings = None
        if embedding_path:
            self.embeddings = torch.load(
                embedding_path, map_location=torch.device("cpu")
            )

    def encode_documents(self, texts):
        """
        Encode list of documents into embeddings (numpy)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def encode_query(self, query: str):
        """
        Encode single query into embedding (numpy)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def get_precomputed_embeddings(self):
        """
        Returns preloaded document embeddings (if available)
        """
        if self.embeddings is None:
            raise ValueError("No precomputed embeddings loaded.")
        return self.embeddings.numpy()
