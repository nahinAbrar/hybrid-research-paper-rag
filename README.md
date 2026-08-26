# Hybrid Multimodal RAG for Research Paper Question Answering

## Project Goal
A Retrieval-Augmented Generation (RAG) system that answers questions about a single uploaded research paper (PDF) by retrieving and reasoning over text, figures, and tables, using hybrid retrieval (BM25 + dense vectors) and a multimodal LLM.

## Architecture Diagram
```
[PDF Upload]
     │
     ▼
[Parsing Module] ───(Extracts Text, Tables, Figures)
     │
     ▼
[Hybrid Retrieval] ◄──(BM25 Search) + (Vector Search)
     │
     ├── Reciprocal Rank Fusion (RRF)
     ▼
[Multimodal LLM] ───(Gemini API)
     │
     ▼
[Answer + Evidence]
```

## Tech Stack
| Component | Technology |
|---|---|
| PDF parsing | docling, pymupdf (fitz) |
| Chunking | langchain text splitters |
| Keyword retrieval | rank_bm25 |
| Embeddings | sentence-transformers (bge-small-en or specter2) |
| Vector store | chromadb |
| Fusion | Reciprocal Rank Fusion (custom function, no extra library) |
| Multimodal LLM | Google Gemini API free tier, via the google-genai SDK |
| Backend | fastapi + uvicorn |
| Demo UI | streamlit |
| Eval/analysis | pandas + plain csv/json logs |

## Team & Ownership

* **Member 1 (Parsing):** `parsing/` module (PDF -> structured chunks)
* **Member 2 (Retrieval):** `retrieval/` module (BM25 + vector + hybrid fusion)
* **Member 3 (Multimodal):** `multimodal/` module (Evidence linking + Gemini calls)
* **Member 4 (Eval/App):** `eval/` and `app/` modules (Experiments, metrics, Streamlit UI)
