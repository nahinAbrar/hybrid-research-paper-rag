"""
Keyword retrieval index using rank_bm25.
"""
from typing import List, Dict, Any

class BM25Retriever:
    """
    A retriever that uses BM25 for keyword-based search over paper chunks.
    """
    def __init__(self):
        """Initializes the BM25 retriever."""
        # TODO: implement
        pass

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of parsed chunks for BM25 retrieval.

        Args:
            chunks: A list of dictionaries representing the chunks to index.
        """
        # TODO: implement
        raise NotImplementedError("BM25 indexing is not yet implemented.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the given query.

        Args:
            query: The search query string.
            top_k: The number of top results to return.

        Returns:
            A list of the top_k matching chunks with their BM25 scores.
        """
        # TODO: implement
        raise NotImplementedError("BM25 search is not yet implemented.")
