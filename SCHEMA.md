# Parsing Output Schema

This document defines the exact JSON structure that the `parsing/` module will output per paper.
The output should be a JSON list of chunk objects. Each chunk object should conform to the structure below.

## Chunk Object Structure
- `id` (string): Unique identifier for the chunk.
- `type` (string): Type of the chunk, which must be one of `text`, `table`, or `figure`.
- `page` (integer): Page number where the chunk is located (1-indexed).
- `section` (string): The title of the section containing the chunk.
- `text` (string): The parsed content (text) of the chunk. For tables, this could be markdown or CSV representation.
- `caption` (string, optional): Caption for figures or tables. Null for text chunks.
- `nearby_text` (string, optional): Text located before/after the figure or table for context.
- `image_path` (string, optional): Local path to the extracted image file for figures or tables.

## Example Output

```json
[
  {
    "id": "chunk-1",
    "type": "text",
    "page": 1,
    "section": "Abstract",
    "text": "This paper presents a novel approach to...",
    "caption": null,
    "nearby_text": null,
    "image_path": null
  },
  {
    "id": "chunk-2",
    "type": "table",
    "page": 3,
    "section": "Results",
    "text": "| Model | Accuracy |\n|---|---|\n| Ours | 95% |",
    "caption": "Table 1: Main experimental results.",
    "nearby_text": "As seen in Table 1, our model significantly outperforms...",
    "image_path": "data/processed/paper-1/table-1.png"
  },
  {
    "id": "chunk-3",
    "type": "figure",
    "page": 4,
    "section": "Methodology",
    "text": "",
    "caption": "Figure 2: Architecture diagram of the proposed system.",
    "nearby_text": "Figure 2 illustrates the overall architecture...",
    "image_path": "data/processed/paper-1/figure-2.png"
  }
]
```
