"""
Pydantic models for the parsed paper chunks, matching SCHEMA.md.
"""
from typing import Optional, Literal
from pydantic import BaseModel

class Chunk(BaseModel):
    """
    Base schema for a parsed chunk from a research paper.
    """
    id: str
    type: Literal["text", "table", "figure"]
    page: int
    section: str
    text: str
    caption: Optional[str] = None
    nearby_text: Optional[str] = None
    image_path: Optional[str] = None
