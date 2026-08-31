"""
Reciprocal Rank Fusion (RRF) for combining BM25 and Vector search results.
"""
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines results from BM25 and Vector search using Reciprocal Rank Fusion.

    Args:
        bm25_results: Ranked list of results from BM25.
        vector_results: Ranked list of results from Vector search.
        k: The RRF constant.

    Returns:
        A single fused and ranked list of results.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    
    # Process BM25 results
    for rank, chunk in enumerate(bm25_results, start=1):
        chunk_id = chunk["id"]
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk.copy()
            rrf_scores[chunk_id] = 0.0
            
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        
    # Process Vector results
    for rank, chunk in enumerate(vector_results, start=1):
        chunk_id = chunk["id"]
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk.copy()
            rrf_scores[chunk_id] = 0.0
            
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        
    # Sort by fused score
    fused_results = []
    for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        fused_chunk = chunk_map[chunk_id]
        fused_chunk["fusion_score"] = score
        # Optionally remove the individual scores to keep it clean
        fused_chunk.pop("score", None) 
        fused_results.append(fused_chunk)
        
    return fused_results
