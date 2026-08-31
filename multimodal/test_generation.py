"""
Test script for verifying the Multimodal Generation (M3) module.
"""
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import BM25Retriever
from retrieval.vector_index import VectorRetriever
from retrieval.hybrid_fusion import reciprocal_rank_fusion
from multimodal.evidence_linker import link_evidence
from multimodal.llm_client import GeminiMultimodalClient

def main():
    # Force UTF-8 encoding for Windows console printing
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment or .env file.")
        print("Please add it to .env before running this test.")
        return

    chunks_path = project_root / "data" / "processed" / "chunks.json"
    chroma_dir = project_root / "data" / "processed" / "chroma_db"
    
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found.")
        return
        
    # Load chunks
    print("Loading chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # Initialize Retrievers
    print("\n--- Initializing Retrievers & Retrieving Context ---")
    bm25 = BM25Retriever()
    vector = VectorRetriever(persist_directory=str(chroma_dir))
    
    bm25.index_chunks(chunks)
    vector.index_chunks(chunks)
    
    query = "How is SYCON-Bench used to measure sycophancy? Please include details from any tables or figures if relevant."
    print(f"\nQuery: '{query}'")
    
    deep_bm25 = bm25.search(query, top_k=60)
    deep_vector = vector.search(query, top_k=60)
    fused_res = reciprocal_rank_fusion(deep_bm25, deep_vector)[:5]
    
    print("\n--- Linking Visual Evidence ---")
    enriched_chunks = link_evidence(fused_res, chunks)
    for i, chunk in enumerate(enriched_chunks, start=1):
        has_img = bool(chunk.get("image_path") or chunk.get("linked_images"))
        print(f" [{i}] {chunk['type'].upper()} (Page {chunk['page']}) - Has attached images: {has_img}")
        
    print("\n--- Calling Gemini Multimodal LLM ---")
    client = GeminiMultimodalClient(model_name="gemini-3.5-flash")
    
    try:
        answer = client.generate_answer(query, enriched_chunks)
        print("\n================ FINAL ANSWER ================\n")
        print(answer)
        print("\n==============================================")
    except Exception as e:
        print(f"\nAPI Error: {e}")

if __name__ == "__main__":
    main()
