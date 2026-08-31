"""
Dense vector retrieval index using sentence-transformers and chromadb.
"""
import json
import uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class VectorRetriever:
    """
    A retriever that uses sentence-transformers for generating embeddings
    and ChromaDB for vector storage and similarity search.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", persist_directory: str = "./chroma_db"):
        """
        Initializes the Vector retriever.

        Args:
            model_name: The HuggingFace model name for sentence embeddings.
            persist_directory: Path to persist the ChromaDB database.
        """
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create a collection for our chunks
        self.collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Combines relevant text fields from a chunk for vector embeddings."""
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
        Generates embeddings for the chunks and stores them in ChromaDB.

        Args:
            chunks: A list of dictionaries representing the chunks to index.
        """
        if not chunks:
            return

        texts = [self._get_chunk_text(c) for c in chunks]
        
        # We need IDs for chromadb
        ids = [c.get("id", str(uuid.uuid4())) for c in chunks]
        
        # Store the entire chunk as a JSON string in metadata so we can recover it later
        metadatas = [{"chunk_data": json.dumps(c)} for c in chunks]

        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        print("Adding embeddings to ChromaDB...")
        # Batch insert into ChromaDB to avoid size limits (though 300 chunks is tiny)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print("Vector indexing complete.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search for the given query.

        Args:
            query: The search query string.
            top_k: The number of top results to return.

        Returns:
            A list of the top_k matching chunks with their similarity scores.
        """
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        scored_chunks = []
        if not results["metadatas"] or not results["metadatas"][0]:
            return scored_chunks
            
        for i in range(len(results["metadatas"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Reconstruct the chunk from metadata
            chunk = json.loads(meta["chunk_data"])
            
            # For cosine distance in ChromaDB, lower is closer to 0. 
            # We convert to a similarity score (1 - distance) for easier fusion logic.
            chunk["score"] = 1.0 - float(distance)
            scored_chunks.append(chunk)
            
        return scored_chunks
