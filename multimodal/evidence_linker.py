"""
Attaches relevant figures/tables to retrieved text chunks for the LLM context.
"""
import re
from typing import List, Dict, Any

def link_evidence(retrieved_chunks: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enriches retrieved text chunks by linking nearby or referenced figures and tables.

    Args:
        retrieved_chunks: The top chunks returned by the hybrid retrieval system.
        all_chunks: All available chunks from the parsed paper.

    Returns:
        A list of retrieved chunks, potentially augmented with linked figure/table metadata and paths.
    """
    enriched_chunks = []
    
    # Pre-index image chunks for fast lookup
    image_chunks = [c for c in all_chunks if c.get("image_path")]
    
    for chunk in retrieved_chunks:
        enriched_chunk = chunk.copy()
        
        # If chunk already has an image, nothing to link
        if enriched_chunk.get("image_path"):
            enriched_chunks.append(enriched_chunk)
            continue
            
        text = enriched_chunk.get("text", "")
        linked_images = []
        
        # Simple regex to find "Figure X" or "Table Y"
        fig_refs = re.findall(r'(?i)(?:figure|fig\.?)\s*(\d+)', text)
        tab_refs = re.findall(r'(?i)table\s*(\d+)', text)
        
        for ic in image_chunks:
            caption = ic.get("caption", "") or ""
            ctype = ic.get("type", "")
            
            # Check if this image chunk matches a referenced figure
            if ctype == "figure" and any(re.search(rf'(?i)(?:figure|fig\.?)\s*{ref}', caption) for ref in fig_refs):
                linked_images.append(ic["image_path"])
                
            # Check if this image chunk matches a referenced table
            if ctype == "table" and any(re.search(rf'(?i)table\s*{ref}', caption) for ref in tab_refs):
                linked_images.append(ic["image_path"])
                
        # Deduplicate and attach
        if linked_images:
            # We add a custom field for the LLM client to parse
            enriched_chunk["linked_images"] = list(set(linked_images))
            
        enriched_chunks.append(enriched_chunk)
        
    return enriched_chunks
