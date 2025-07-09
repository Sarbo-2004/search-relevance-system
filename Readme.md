# 🔍 Unified Search + Recommendation System

A full-stack **Search and Recommendation System**, inspired by real-world systems used by Google, Amazon, and Netflix.

🧠 Powered by NLP pipelines, SBERT/BM25 retrieval, and optional LLM-based query rewriting. Built to showcase end-to-end real-world search systems.

---

## 🚀 Features

- ✅ **Search Relevance System** with:
  - TF-IDF / BM25 / SBERT retrieval
  - Optional: LLM-powered smart query rewriting
  - Scoring based on semantic similarity and basic rules

- 🔁 **Recommendation Support** (coming soon):
  - Related or personalized product suggestions

- 💡 Modular and beginner-friendly for learning and showcasing

---

## 📦 Project Structure

```
.
├── data/                    # Datasets and embeddings
│   ├── df_nlp.pkl           # Preprocessed product data
│   └── sbert_product_embeddings.pt
├── embeddings/             # SBERT embedding generation
│   └── generate_sbert_embeddings.py
├── query_rewriting/        # LLM-powered rewriting module
│   └── llm_rewriter.py
├── ranking/                # (Currently skipped)
│   └── simple_ranker.py     # Optional rule-based fallback
├── retriever/
│   ├── bm25.py              # BM25 search
│   └── sbert.py             # SBERT-based search
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter experiments (optional)
├── app/                    # (Optional) UI with Streamlit or FastAPI
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/search-relevance-recommender.git
cd search-relevance-recommender

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. ▶️ Run the app
```
uvicorn api.main:app --reload
```
Visit: http://localhost:8000/docs

---

## 🧠 Technologies Used

- Python, Pandas, NumPy
- NLP: `nltk`, `scikit-learn`, `sentence-transformers`
- Retrieval: `rank-bm25`, `sentence-transformers`
- Optional: `openai` for query rewriting
- Interface: Streamlit / FastAPI (optional)

---

## 📈 Future Additions

- ✅ Add hybrid scoring (BM25 + SBERT)
- 🔄 Personalized recommender module
- 🔁 Add Learning-to-Rank (ML-based reranking) — coming soon
- 🌐 Build REST API for deployment

---

## 📄 License

MIT License. Free to use for learning, research, or personal projects.

---

## 🙋‍♂️ Author

**Sarbojeet Soe Mondal**  
Feel free to connect on [LinkedIn](www.linkedin.com/in/sarbojeet-soe-mondal-data-scientist) or raise issues for contributions!
