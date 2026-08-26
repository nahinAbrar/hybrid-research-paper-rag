"""
Pipeline for parsing a PDF research paper into structured chunks using Docling and PyMuPDF.
"""
from typing import List
from pathlib import Path
from parsing.schema import Chunk

def parse_pdf_to_chunks(pdf_path: str | Path, output_dir: str | Path) -> List[Chunk]:
    """
    Parses a PDF file and extracts text, tables, and figures into structured chunks.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where extracted images (tables/figures) will be saved.

    Returns:
        A list of Chunk objects representing the parsed content.
    """
    # TODO: implement
    raise NotImplementedError("PDF parsing logic is not yet implemented.")
