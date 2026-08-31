"""
Pipeline for parsing a PDF research paper into structured chunks using Docling and PyMuPDF.
"""
import uuid
import json
import pymupdf as fitz  # PyMuPDF (modern import)
import sys
from typing import List
from pathlib import Path

# Add project root to sys.path so 'parsing' module can be imported when running as a standalone script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from parsing.schema import Chunk

# Import Docling if available
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.document import SectionHeaderItem, TextItem, TableItem, PictureItem
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


def _crop_image_from_pdf(pdf_path: str, page_num: int, bbox, output_path: str) -> str:
    """
    Crops an image from the PDF using PyMuPDF given a bounding box.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1] # 0-indexed in PyMuPDF
        
        # Bbox formats might differ slightly; fitz Rect takes (x0, y0, x1, y1)
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        pix = page.get_pixmap(clip=rect, dpi=150)
        pix.save(output_path)
        doc.close()
        return output_path
    except Exception as e:
        print(f"Failed to extract image from {pdf_path}, page {page_num}: {e}")
        return None


def _get_caption_text(item, doc):
    if not hasattr(item, 'captions') or not item.captions:
        return None
    cap = item.captions[0]
    if hasattr(cap, 'text'):
        return cap.text
    
    try:
        # In Docling v2, captions are RefItems which need to be resolved against the document
        if hasattr(cap, 'resolve'):
            resolved_node = cap.resolve(doc)
            if hasattr(resolved_node, 'text'):
                return resolved_node.text
    except Exception:
        pass
    
    return None


def parse_pdf_to_chunks(pdf_path: str | Path, output_dir: str | Path) -> List[Chunk]:
    """
    Parses a PDF file and extracts text, tables, and figures into structured chunks.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where extracted images (tables/figures) will be saved.

    Returns:
        A list of Chunk objects representing the parsed content.
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is required. Please install it via 'pip install docling'.")

    pdf_path_str = str(pdf_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Configure Docling to extract images for tables and figures natively if possible
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    
    # Initialize Document Converter
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print(f"Parsing document: {pdf_path_str}")
    result = converter.convert(pdf_path_str)
    doc = result.document

    chunks: List[Chunk] = []
    current_section = "General"
    
    # Store text elements sequentially to allow extracting "nearby_text" for context
    text_buffer = []

    for item, level in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            current_section = item.text
            text_buffer.append(item.text)
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                type="text",
                page=item.prov[0].page_no if item.prov else 1,
                section=current_section,
                text=item.text
            )
            chunks.append(chunk)

        elif isinstance(item, TextItem):
            text_buffer.append(item.text)
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                type="text",
                page=item.prov[0].page_no if item.prov else 1,
                section=current_section,
                text=item.text
            )
            chunks.append(chunk)

        elif isinstance(item, TableItem):
            table_text = item.export_to_markdown(doc=doc) if hasattr(item, 'export_to_markdown') else "Table extracted."
            page_no = item.prov[0].page_no if item.prov else 1
            
            # Save Table image
            image_path = None
            if hasattr(item, 'get_image') and item.get_image(doc):
                img = item.get_image(doc)
                image_name = f"table_{page_no}_{uuid.uuid4().hex[:8]}.png"
                img_path = output_dir_path / image_name
                img.save(img_path)
                image_path = str(img_path)
            elif item.prov and item.prov[0].bbox:
                # Fallback to PyMuPDF cropping
                bbox = item.prov[0].bbox.as_tuple()
                image_name = f"table_{page_no}_{uuid.uuid4().hex[:8]}.png"
                img_path = output_dir_path / image_name
                image_path = _crop_image_from_pdf(pdf_path_str, page_no, bbox, str(img_path))

            chunk = Chunk(
                id=str(uuid.uuid4()),
                type="table",
                page=page_no,
                section=current_section,
                text=table_text,
                caption=_get_caption_text(item, doc),
                nearby_text="\n".join(text_buffer[-3:]) if len(text_buffer) > 0 else None,
                image_path=image_path
            )
            chunks.append(chunk)

        elif isinstance(item, PictureItem):
            page_no = item.prov[0].page_no if item.prov else 1
            
            # Save Picture image
            image_path = None
            if hasattr(item, 'get_image') and item.get_image(doc):
                img = item.get_image(doc)
                image_name = f"figure_{page_no}_{uuid.uuid4().hex[:8]}.png"
                img_path = output_dir_path / image_name
                img.save(img_path)
                image_path = str(img_path)
            elif item.prov and item.prov[0].bbox:
                # Fallback to PyMuPDF cropping
                bbox = item.prov[0].bbox.as_tuple()
                image_name = f"figure_{page_no}_{uuid.uuid4().hex[:8]}.png"
                img_path = output_dir_path / image_name
                image_path = _crop_image_from_pdf(pdf_path_str, page_no, bbox, str(img_path))

            chunk = Chunk(
                id=str(uuid.uuid4()),
                type="figure",
                page=page_no,
                section=current_section,
                text="",  # Figures usually don't have text bodies
                caption=_get_caption_text(item, doc),
                nearby_text="\n".join(text_buffer[-3:]) if len(text_buffer) > 0 else None,
                image_path=image_path
            )
            chunks.append(chunk)

    print(f"Successfully extracted {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    # Example local usage for testing the M1 Pipeline
    import sys
    
    # We use project root based on __file__ to make paths robust
    project_root = Path(__file__).resolve().parent.parent
    
    if len(sys.argv) > 1:
        sample_pdf = Path(sys.argv[1])
    else:
        sample_pdf = project_root / "data" / "papers" / "sample.pdf"
        
    output_images = project_root / "data" / "processed" / "images"
    output_json = project_root / "data" / "processed" / "chunks.json"
    
    if sample_pdf.exists():
        parsed_chunks = parse_pdf_to_chunks(sample_pdf, output_images)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in parsed_chunks], f, indent=2)
        print(f"Saved JSON output to {output_json}")
    else:
        print(f"Please provide a valid PDF path or place a PDF at {sample_pdf} to test the script.")
