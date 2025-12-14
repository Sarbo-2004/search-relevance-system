from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.prompts import RAG_PROMPT


def retrieve_docs(inputs):
    """
    LCEL-compatible retriever function
    """
    query = inputs["question"]
    hybrid = inputs["hybrid"]
    encoder = inputs["encoder"]
    top_k = inputs.get("top_k", 5)

    query_embedding = encoder.encode_query(query)

    df = hybrid.search(
        query,
        query_embedding,
        top_k=top_k
    )

    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["search_text"],
                metadata={
                    "title": row.get("title"),
                    "source": row.get("retrieval_source")
                }
            )
        )
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(hybrid_retriever, encoder, top_k=5):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    retriever_runnable = RunnableLambda(
        lambda question: retrieve_docs({
            "question": question,
            "hybrid": hybrid_retriever,
            "encoder": encoder,
            "top_k": top_k
        })
    )

    rag_chain = (
        {
            "context": retriever_runnable | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
    )

    return rag_chain
