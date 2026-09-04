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

## How to Run (Step-by-Step)

### Step 1 — Parse the PDF (M1)
Place your research paper PDF inside the `data/papers/` folder, then run:
```powershell
.\venv\Scripts\python.exe parsing\parse_pdf.py
```
**What it does:** Uses Docling to extract text, tables, and figures from the PDF. Saves structured chunks to `data/processed/chunks.json` and cropped images to `data/processed/images/`.

### Step 2 — Test Retrieval (M2) *(optional)*
```powershell
.\venv\Scripts\python.exe retrieval\test_retrieval.py
```
**What it does:** Indexes all chunks into BM25 and ChromaDB, runs a sample query, and prints the top results from BM25, Vector Search, and Hybrid Fusion side by side. Use this to verify the retrieval pipeline is working correctly.

### Step 3 — Test Generation (M3) *(optional)*
```powershell
.\venv\Scripts\python.exe multimodal\test_generation.py
```
**What it does:** Runs the full RAG pipeline end-to-end in the terminal. It retrieves relevant chunks, links any referenced figures/tables, and sends everything (text + images) to the Gemini API. Prints the final LLM-generated answer with citations.

### Step 4 — Run Evaluation Experiments (M4) *(optional)*
```powershell
.\venv\Scripts\python.exe eval\run_experiments.py
```
**What it does:** Runs a suite of test queries through BM25 Only, Vector Only, and Hybrid Fusion, and compares them using Recall@5 and MRR metrics. Outputs a table proving that Hybrid Fusion outperforms standalone methods.

### Step 5 — Launch the Streamlit Demo UI (M4)
```powershell
.\venv\Scripts\streamlit.exe run app\streamlit_app.py
```
**What it does:** Opens an interactive web app in your browser (usually at `http://localhost:8501`). You can type questions about the paper and see the multimodal answer alongside the retrieved text chunks and images.

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
