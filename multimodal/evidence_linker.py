"""
Attaches relevant figures/tables to retrieved text chunks for the LLM context.
"""
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
    # TODO: implement
    raise NotImplementedError("Evidence linking is not yet implemented.")
