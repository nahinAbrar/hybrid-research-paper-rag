"""
Wrapper for the Google Gemini API (via google-genai SDK) to perform multimodal reasoning.
"""
from typing import List, Dict, Any

class GeminiMultimodalClient:
    """Client for interacting with the Google Gemini Multimodal API."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the Gemini client.

        Args:
            api_key: The Google Gemini API key. If None, expects GEMINI_API_KEY env variable.
            model_name: The specific Gemini model to use.
        """
        # TODO: implement
        pass

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer to the user's query based on the provided retrieved chunks.
        The client should handle uploading linked images to the Gemini API if figures/tables are in context.

        Args:
            query: The user's question.
            context_chunks: The retrieved chunks (text, tables, figures) to use as context.

        Returns:
            The generated string answer from the LLM.
        """
        # TODO: implement
        raise NotImplementedError("LLM answer generation is not yet implemented.")
