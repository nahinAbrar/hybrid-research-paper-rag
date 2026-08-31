"""
Wrapper for the Google Gemini API (via google-genai SDK) to perform multimodal reasoning.
"""
import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
import PIL.Image

class GeminiMultimodalClient:
    """Client for interacting with the Google Gemini Multimodal API."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-3.5-flash"):
        """
        Initializes the Gemini client.

        Args:
            api_key: The Google Gemini API key. If None, expects GEMINI_API_KEY env variable.
            model_name: The specific Gemini model to use.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment.")
            
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

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
        context_text = "Retrieved Context:\n"
        images = []
        
        for i, chunk in enumerate(context_chunks, start=1):
            context_text += f"\n[{i}] Page {chunk.get('page')}, Section: {chunk.get('section', 'N/A')}\n"
            if chunk.get('text'):
                context_text += f"{chunk['text']}\n"
            if chunk.get('caption'):
                context_text += f"Caption: {chunk['caption']}\n"
                
            # Collect direct images (if the chunk is a figure/table)
            if chunk.get("image_path") and os.path.exists(chunk["image_path"]):
                try:
                    images.append(PIL.Image.open(chunk["image_path"]))
                except Exception as e:
                    print(f"Warning: Failed to load image {chunk['image_path']}: {e}")
                    
            # Collect linked images (from evidence_linker)
            if chunk.get("linked_images"):
                for img_path in chunk["linked_images"]:
                    if os.path.exists(img_path):
                        try:
                            images.append(PIL.Image.open(img_path))
                        except Exception as e:
                            print(f"Warning: Failed to load linked image {img_path}: {e}")

        prompt = f"{context_text}\n\nQuestion: {query}"
        
        # Mix images and text for the multimodal payload
        contents = images + [prompt]
        
        sys_instruction = (
            "You are a highly capable AI research assistant. You are provided with text and visual evidence "
            "from a research paper. Your task is to answer the user's question accurately.\n\n"
            "CRITICAL INSTRUCTION - ANOMALY DETECTION: Before answering, first closely compare any provided "
            "figures/tables against their text captions or the surrounding text. If the caption contradicts "
            "what the visual figure actually portrays (e.g. caption says linear but graph is exponential), "
            "you MUST flag this anomaly to the user explicitly, and ALWAYS trust the visual data in the figure "
            "over the text caption.\n\n"
            "Finally, provide clear, accurate answers citing the provided context numbers (e.g., [1], [2])."
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.2,
            )
        )
        
        return response.text
