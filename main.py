import streamlit as st
import requests

st.set_page_config(page_title="Search Relevance System", layout="centered")

API_URL = "http://localhost:8000/search"

st.title("🔍 Search Relevance System")
st.write("Enter a product query to test this search engine.")
st.write("Note: Please search for products based on electronics items only ")

# --- User input ---
query = st.text_input("Enter your search query")

if query:
    with st.spinner("Searching..."):
        response = requests.get(API_URL, params={"query": query})
        
        if response.status_code == 200:
            results = response.json()
            st.success(f"Top {len(results)} results for: **{query}**")
            
            for i, item in enumerate(results, 1):
                st.markdown(f"**{i}. {item['title']}**")
                st.markdown(f"Price: ${item['price']}")
                st.markdown(f"Description: {item['search_text'][:200]}{'...' if len(item['search_text']) > 200 else ''}")
                st.markdown("---")
        else:
            st.error("❌ Failed to fetch results. Check if FastAPI server is running.")

