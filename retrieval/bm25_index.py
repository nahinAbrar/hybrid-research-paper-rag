"""
Keyword retrieval index using rank_bm25.
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    A retriever that uses BM25 for keyword-based search over paper chunks.
    """
    def __init__(self):
        """Initializes the BM25 retriever."""
        self.bm25 = None
        self.chunks = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split() if text else []

    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Combines relevant text fields from a chunk for indexing."""
        parts = []
        if chunk.get("section"):
            parts.append(chunk["section"])
        if chunk.get("text"):
            parts.append(chunk["text"])
        if chunk.get("caption"):
            parts.append(chunk["caption"])
        return " ".join(parts)

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of parsed chunks for BM25 retrieval.

        Args:
            chunks: A list of dictionaries representing the chunks to index.
        """
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(self._get_chunk_text(c)) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the given query.

        Args:
            query: The search query string.
            top_k: The number of top results to return.

        Returns:
            A list of the top_k matching chunks with their BM25 scores.
        """
        if not self.bm25:
            raise ValueError("BM25 index is empty. Call index_chunks() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair chunks with their scores
        scored_chunks = []
        for idx, score in enumerate(scores):
            # We only care about positive scores
            if score > 0:
                chunk_copy = self.chunks[idx].copy()
                chunk_copy["score"] = float(score)
                scored_chunks.append(chunk_copy)
                
        # Sort by score descending and return top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
