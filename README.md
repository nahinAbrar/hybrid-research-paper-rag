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
| PDF parsing | docling, pymupdf |
| Keyword retrieval | rank_bm25 |
| Embeddings | sentence-transformers (bge-small-en-v1.5) |
| Vector store | chromadb |
| Fusion | Reciprocal Rank Fusion (custom implementation) |
| Multimodal LLM | Google Gemini API (gemini-3.5-flash) via google-genai SDK |
| Demo UI | streamlit |
| Eval/analysis | pandas, custom metrics (Recall@k, MRR) |

---

## Setup & Installation

### 1. Clone and create virtual environment
```bash
git clone https://github.com/nahinAbrar/hybrid-research-paper-rag.git
cd hybrid-research-paper-rag
python -m venv venv
```

### 2. Activate the virtual environment
```powershell
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/) and click **"Get API key"**.
2. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your key:
   ```
   GEMINI_API_KEY=AIzaSyA_your_key_here
   ```

---

## How to Run

### Quick Start — Launch the Demo UI
```powershell
.\venv\Scripts\streamlit.exe run app\streamlit_app.py
```
This opens a web app at `http://localhost:8501`. From there:
1. **Upload** any research paper PDF using the sidebar
2. **Watch** the pipeline process the paper in real-time (parsing → indexing → evaluation)
3. **Chat** with the paper — ask questions and view the multimodal answer with retrieved evidence

> **Note:** The app handles everything automatically — no need to run any scripts beforehand. You can upload different papers and the system resets cleanly each time.

---

### Developer Tools *(optional, for testing individual modules)*

#### Parse a PDF manually (M1)
```powershell
.\venv\Scripts\python.exe parsing\parse_pdf.py
```
Uses Docling to extract text, tables, and figures. Saves chunks to `data/processed/chunks.json` and images to `data/processed/images/`.

#### Test Retrieval (M2)
```powershell
.\venv\Scripts\python.exe retrieval\test_retrieval.py
```
Indexes chunks and runs a sample query through BM25, Vector Search, and Hybrid Fusion side by side.

#### Test Generation (M3)
```powershell
.\venv\Scripts\python.exe multimodal\test_generation.py
```
Runs the full RAG pipeline in the terminal: retrieves chunks, links figures, and generates a Gemini-powered answer.

#### Run Evaluation Experiments (M4)
```powershell
.\venv\Scripts\python.exe eval\run_experiments.py
```
Compares BM25 Only vs Vector Only vs Hybrid Fusion using Recall@5 and MRR metrics.

---

## Project Structure
```
hybrid-research-paper-rag/
├── parsing/              # M1: PDF parsing and chunking
│   ├── parse_pdf.py      #   Main parsing script (run this first)
│   └── schema.py         #   Chunk data model
├── retrieval/            # M2: Hybrid retrieval
│   ├── bm25_index.py     #   BM25 keyword search
│   ├── vector_index.py   #   Dense vector search (ChromaDB)
│   ├── hybrid_fusion.py  #   Reciprocal Rank Fusion (RRF)
│   └── test_retrieval.py #   Retrieval test harness
├── multimodal/           # M3: Multimodal generation
│   ├── evidence_linker.py#   Links figures/tables to text chunks
│   ├── llm_client.py     #   Gemini API wrapper
│   └── test_generation.py#   Generation test harness
├── eval/                 # M4: Evaluation
│   ├── metrics.py        #   Recall@k, MRR metrics
│   └── run_experiments.py#   A/B/C experiment runner
├── app/                  # M4: Demo UI
│   └── streamlit_app.py  #   Streamlit chat interface
├── data/
│   ├── papers/           #   Place your PDF here
│   ├── processed/        #   Auto-generated chunks + images
│   └── qa_dataset.json   #   Evaluation question set
├── .env.example          #   API key template
├── requirements.txt      #   Python dependencies
└── README.md             #   This file
```

## Team & Ownership

* **Member 1 (Parsing):** `parsing/` module (PDF -> structured chunks)
* **Member 2 (Retrieval):** `retrieval/` module (BM25 + vector + hybrid fusion)
* **Member 3 (Multimodal):** `multimodal/` module (Evidence linking + Gemini calls)
* **Member 4 (Eval/App):** `eval/` and `app/` modules (Experiments, metrics, Streamlit UI)
