"""
Test script for verifying the Retrieval (M2) module.
"""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import BM25Retriever
from retrieval.vector_index import VectorRetriever
from retrieval.hybrid_fusion import reciprocal_rank_fusion

def main():
    # Force UTF-8 encoding for Windows console printing
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "processed" / "chunks.json"
    chroma_dir = project_root / "data" / "processed" / "chroma_db"
    
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Please run the parsing pipeline first.")
        return
        
    # Load chunks
    print("Loading chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks.")
    
    # Initialize Retrievers
    print("\n--- Initializing Retrievers ---")
    bm25 = BM25Retriever()
    vector = VectorRetriever(persist_directory=str(chroma_dir))
    
    # Index
    print("\n--- Indexing Chunks ---")
    bm25.index_chunks(chunks)
    vector.index_chunks(chunks)
    
    # Test Query
    query = "How is SYCON-Bench used to measure sycophancy?"
    print(f"\n--- Testing Query: '{query}' ---")
    
    print("\n[ BM25 Results ]")
    bm25_res = bm25.search(query, top_k=5)
    for i, r in enumerate(bm25_res):
        print(f" {i+1}. [Score: {r['score']:.4f}] {r['type'].upper()} (Page {r['page']}): {r['text'][:100]}...")
        
    print("\n[ Vector Results ]")
    vector_res = vector.search(query, top_k=5)
    for i, r in enumerate(vector_res):
        print(f" {i+1}. [Score: {r['score']:.4f}] {r['type'].upper()} (Page {r['page']}): {r['text'][:100]}...")
        
    print("\n[ Hybrid Fusion Results ]")
    # Note: For best results, we usually fuse deeper lists (e.g. top 60 of each) to get a robust top 5
    deep_bm25 = bm25.search(query, top_k=60)
    deep_vector = vector.search(query, top_k=60)
    fused_res = reciprocal_rank_fusion(deep_bm25, deep_vector)[:5]
    
    for i, r in enumerate(fused_res):
        print(f" {i+1}. [Fusion Score: {r['fusion_score']:.4f}] {r['type'].upper()} (Page {r['page']}): {r['text'][:100]}...")

if __name__ == "__main__":
    main()
