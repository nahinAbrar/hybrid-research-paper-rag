"""
Streamlit web application for the interactive RAG demo.
"""
import sys
import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import BM25Retriever
from retrieval.vector_index import VectorRetriever
from retrieval.hybrid_fusion import reciprocal_rank_fusion
from multimodal.evidence_linker import link_evidence
from multimodal.llm_client import GeminiMultimodalClient

st.set_page_config(page_title="Hybrid RAG Chat", layout="wide")

@st.cache_resource
def initialize_system():
    """Loads the indices and Gemini client into memory."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    
    chunks_path = project_root / "data" / "processed" / "chunks.json"
    chroma_dir = project_root / "data" / "processed" / "chroma_db"
    
    if not chunks_path.exists():
        return None, None, None, "Chunks not found. Run parsing first."
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    bm25 = BM25Retriever()
    vector = VectorRetriever(persist_directory=str(chroma_dir))
    
    bm25.index_chunks(chunks)
    vector.index_chunks(chunks)
    
    try:
        # Changed to 3.5-flash for 2026 compatibility
        client = GeminiMultimodalClient(model_name="gemini-3.5-flash")
    except ValueError as e:
        return None, None, None, str(e)
        
    return bm25, vector, client, chunks

def main():
    st.title("📚 Hybrid Multimodal RAG Demo")
    st.markdown("Ask questions about the research paper. The system will retrieve text and visual evidence and answer multimodally.")
    
    bm25, vector, client, chunks = initialize_system()
    
    if chunks is None:
        st.error(chunks) # Prints the error string
        return
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "evidence" in msg:
                with st.expander("View Retrieved Evidence"):
                    for i, chunk in enumerate(msg["evidence"], 1):
                        st.markdown(f"**[{i}] Page {chunk.get('page')} ({chunk.get('type')})**")
                        if chunk.get('text'):
                            st.caption(chunk['text'][:300] + "...")
                        if chunk.get('image_path') and os.path.exists(chunk['image_path']):
                            st.image(chunk['image_path'], caption="Direct Image")
                        if chunk.get('linked_images'):
                            for img in chunk['linked_images']:
                                if os.path.exists(img):
                                    st.image(img, caption="Linked Visual Evidence")
                                    
    # Chat Input
    if query := st.chat_input("What is the paper about?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                # 1. Retrieve
                deep_bm25 = bm25.search(query, top_k=60)
                deep_vector = vector.search(query, top_k=60)
                
                # 2. Fuse
                fused_res = reciprocal_rank_fusion(deep_bm25, deep_vector)[:5]
                
                # 3. Link Evidence
                enriched_chunks = link_evidence(fused_res, chunks)
                
                # 4. Generate
                try:
                    answer = client.generate_answer(query, enriched_chunks)
                    st.markdown(answer)
                    
                    with st.expander("View Retrieved Evidence"):
                        for i, chunk in enumerate(enriched_chunks, 1):
                            st.markdown(f"**[{i}] Page {chunk.get('page')} ({chunk.get('type')})**")
                            if chunk.get('text'):
                                st.caption(chunk['text'][:300] + "...")
                            if chunk.get('image_path') and os.path.exists(chunk['image_path']):
                                st.image(chunk['image_path'], caption="Direct Image")
                            if chunk.get('linked_images'):
                                for img in chunk['linked_images']:
                                    if os.path.exists(img):
                                        st.image(img, caption="Linked Visual Evidence")
                                        
                    st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": enriched_chunks})
                except Exception as e:
                    st.error(f"API Error: {e}")

if __name__ == "__main__":
    main()
