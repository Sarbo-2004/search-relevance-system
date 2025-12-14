from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from retrieval.sbert import SBERTEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25 import BM25Retriever
from rag.rag_chain import build_rag_chain

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="AI Search Relevance System",
    description="Hybrid Search + RAG using BM25, FAISS, SBERT, LangChain & Gemini",
    version="1.0"
)

# -------------------------
# Request / Response Models
# -------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ProductResult(BaseModel):
    title: str
    retrieval_source: str


class SearchResponse(BaseModel):
    answer: str
    products: list[ProductResult]


# -------------------------
# Load Everything ONCE
# -------------------------
print("🔄 Loading data and models...")

with open("data/df_nlp.pkl", "rb") as f:
    df = pickle.load(f)

encoder = SBERTEncoder("data/sbert_product_embeddings.pt")

faiss_index = FaissIndex(
    encoder.get_precomputed_embeddings(),
    df
)

bm25 = BM25Retriever(
    df,
    text_column="search_text"
)

hybrid_retriever = HybridRetriever(
    bm25,
    faiss_index
)

rag_chain = build_rag_chain(
    hybrid_retriever,
    encoder
)

print("✅ Backend ready")


# -------------------------
# RAG Search Endpoint
# -------------------------
@app.post("/search-rag", response_model=SearchResponse)
def search_rag(request: SearchRequest):
    """
    Main AI-powered search endpoint
    """

    # Run RAG pipeline
    result = rag_chain.invoke(request.query)

    # Extract products again (clean UI data)
    query_embedding = encoder.encode_query(request.query)
    retrieved_df = hybrid_retriever.search(
        request.query,
        query_embedding,
        top_k=request.top_k
    )

    products = []
    for _, row in retrieved_df.iterrows():
        products.append({
            "title": row["title"],
            "retrieval_source": row["retrieval_source"]
        })

    return {
        "answer": result.content if hasattr(result, "content") else str(result),
        "products": products
    }


# -------------------------
# Health Check
# -------------------------
@app.get("/")
def health():
    return {"status": "AI Search Backend is running"}
