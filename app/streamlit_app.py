"""
Streamlit web application for the interactive RAG demo.
Full pipeline visualization: Upload → Parse → Index → Chat.
"""
import sys
import os
import io
import json
import time
import shutil
import tempfile
import streamlit as st
from pathlib import Path
from contextlib import redirect_stdout
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Set HF_TOKEN for authenticated HuggingFace Hub downloads (faster + higher rate limits)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token  # ensure it's set for subprocesses too

from retrieval.bm25_index import BM25Retriever
from retrieval.vector_index import VectorRetriever
from retrieval.hybrid_fusion import reciprocal_rank_fusion
from multimodal.evidence_linker import link_evidence
from multimodal.llm_client import GeminiMultimodalClient
from eval.metrics import calculate_recall_at_k, calculate_mrr

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Multimodal RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (minimal, works with Streamlit default theme) ──
st.markdown("""
<style>
    .log-output {
        background: #1e1e2e;
        color: #a6e3a1;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.78rem;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #313244;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ──────────────────────────────
def init_session():
    defaults = {
        "pipeline_stage": "upload",    # upload | parsing | indexing | ready
        "chunks": None,
        "bm25": None,
        "vector": None,
        "client": None,
        "messages": [],
        "pipeline_logs": [],
        "paper_name": None,
        "chunk_stats": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar: Upload & Pipeline Status ────────────────────────
with st.sidebar:
    st.title("📚 RAG Pipeline")
    st.caption("Hybrid Multimodal Research Paper QA")
    st.divider()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a Research Paper (PDF)",
        type=["pdf"],
        help="Upload any research paper PDF. The system will parse, index, and prepare it for Q&A.",
    )
    
    if uploaded_file and uploaded_file.name != st.session_state.paper_name:
        st.session_state.pipeline_stage = "uploaded"
        st.session_state.paper_name = uploaded_file.name
        st.session_state.chunks = None
        st.session_state.bm25 = None
        st.session_state.vector = None
        st.session_state.messages = []
        st.session_state.pipeline_logs = []
        st.session_state.chunk_stats = None
        
        # ── Clean up old processed data from previous papers ──
        # Delete old extracted images
        old_images_dir = PROJECT_ROOT / "data" / "processed" / "images"
        if old_images_dir.exists():
            shutil.rmtree(old_images_dir)
            old_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Delete old chunks.json
        old_chunks_json = PROJECT_ROOT / "data" / "processed" / "chunks.json"
        if old_chunks_json.exists():
            old_chunks_json.unlink()
        
        # Note: ChromaDB data is cleared via its API during indexing (not here),
        # because the PersistentClient holds a file lock on Windows.
        
        # Delete old PDF files from papers directory
        old_papers_dir = PROJECT_ROOT / "data" / "papers"
        if old_papers_dir.exists():
            for old_pdf in old_papers_dir.glob("*.pdf"):
                old_pdf.unlink()
    
    # Process button
    if st.session_state.pipeline_stage == "uploaded":
        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            st.session_state.pipeline_stage = "parsing"
            st.rerun()
    
    st.divider()
    
    # Pipeline Status Indicators
    st.markdown("### Pipeline Status")
    
    stages = [
        ("upload", "📄", "Upload PDF"),
        ("parsing", "🔬", "M1: Parse & Extract"),
        ("indexing", "🔗", "M2: Build Indices"),
        ("ready", "✅", "Ready for Q&A"),
    ]
    
    current_stage = st.session_state.pipeline_stage
    stage_order = ["upload", "uploaded", "parsing", "indexing", "eval", "ready"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
    
    for stage_key, icon, label in stages:
        stage_idx = stage_order.index(stage_key)
        if current_idx > stage_idx:
            st.success(f"{icon} {label}", icon="✅")
        elif current_idx == stage_idx or (stage_key == "upload" and current_stage == "uploaded"):
            st.info(f"{icon} {label}", icon="⏳")
        else:
            st.markdown(f"⬜ {icon} {label}")
    
    # Show stats when ready
    if st.session_state.chunk_stats:
        st.divider()
        st.markdown("### 📊 Document Stats")
        stats = st.session_state.chunk_stats
        col1, col2 = st.columns(2)
        col1.metric("Total Chunks", stats["total"])
        col2.metric("Pages", stats["pages"])
        col1.metric("Text", stats["text"])
        col2.metric("Tables", stats["tables"])
        col1.metric("Figures", stats["figures"])


# ─── Main Area ─────────────────────────────────────────────────
st.title("📚 Hybrid Multimodal RAG Demo")
st.caption("Upload a research paper → watch the pipeline process it → ask anything about it")

# ─── Stage: Upload ─────────────────────────────────────────────
if st.session_state.pipeline_stage in ("upload", "uploaded"):
    st.markdown("---")
    
    if st.session_state.pipeline_stage == "upload":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 👈 Upload a PDF to get started
            
            The system will:
            1. **Parse** the paper using AI document understanding (Docling)
            2. **Extract** text, tables, and figures with their images
            3. **Index** everything into BM25 + Vector databases
            4. Let you **chat** with the paper using multimodal RAG
            
            > You'll be able to watch each step happen in real-time!
            """)
    else:
        st.info(f"📄 **{st.session_state.paper_name}** uploaded! Click **🚀 Process Document** in the sidebar to begin.", icon="📄")


# ─── Stage: Parsing (M1) ──────────────────────────────────────
if st.session_state.pipeline_stage == "parsing":
    st.markdown("---")
    
    # Save the uploaded file to a temp location
    temp_dir = PROJECT_ROOT / "data" / "papers"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    output_images = PROJECT_ROOT / "data" / "processed" / "images"
    output_json = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    
    # ── Step 1: Parsing ──
    with st.status("🔬 **Step 1/3 — Parsing PDF with Docling AI**", expanded=True) as status:
        st.write("Initializing the Docling document converter...")
        st.write(f"📄 Processing: `{uploaded_file.name}`")
        
        from parsing.parse_pdf import parse_pdf_to_chunks
        
        log_capture = io.StringIO()
        with redirect_stdout(log_capture):
            parsed_chunks = parse_pdf_to_chunks(str(pdf_path), str(output_images))
        
        logs = log_capture.getvalue()
        if logs:
            st.markdown(f'<div class="log-output">{logs}</div>', unsafe_allow_html=True)
        
        # Convert to dicts and save
        chunk_dicts = [c.model_dump() for c in parsed_chunks]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2)
        
        st.session_state.chunks = chunk_dicts
        
        # Compute stats
        text_count = sum(1 for c in chunk_dicts if c["type"] == "text")
        table_count = sum(1 for c in chunk_dicts if c["type"] == "table")
        figure_count = sum(1 for c in chunk_dicts if c["type"] == "figure")
        pages = max(c["page"] for c in chunk_dicts) if chunk_dicts else 0
        
        st.session_state.chunk_stats = {
            "total": len(chunk_dicts),
            "text": text_count,
            "tables": table_count,
            "figures": figure_count,
            "pages": pages,
        }
        
        st.write(f"✅ Extracted **{len(chunk_dicts)}** chunks ({text_count} text, {table_count} tables, {figure_count} figures)")
        status.update(label="✅ **Step 1/3 — PDF Parsed Successfully**", state="complete")
    
    # ── Step 2: Indexing ──
    with st.status("🔗 **Step 2/3 — Building Search Indices**", expanded=True) as status:
        st.write("Building **BM25** keyword index...")
        bm25 = BM25Retriever()
        bm25.index_chunks(chunk_dicts)
        st.write(f"✅ BM25 index ready — {len(chunk_dicts)} documents tokenized")
        
        st.write("Building **Vector** index with `bge-small-en-v1.5` embeddings...")
        chroma_dir = str(PROJECT_ROOT / "data" / "processed" / "chroma_db")
        vector = VectorRetriever(persist_directory=chroma_dir)
        
        # Clear old embeddings from any previous paper
        try:
            vector.client.delete_collection("paper_chunks")
            vector.collection = vector.client.get_or_create_collection(
                name="paper_chunks", metadata={"hnsw:space": "cosine"}
            )
            st.write("🗑️ Cleared old index data")
        except Exception:
            pass
        
        log_capture2 = io.StringIO()
        with redirect_stdout(log_capture2):
            vector.index_chunks(chunk_dicts)
        
        logs2 = log_capture2.getvalue()
        if logs2:
            st.markdown(f'<div class="log-output">{logs2}</div>', unsafe_allow_html=True)
        
        st.session_state.bm25 = bm25
        st.session_state.vector = vector
        
        st.write(f"✅ ChromaDB vector store ready — {len(chunk_dicts)} embeddings stored")
        status.update(label="✅ **Step 2/3 — Indices Built Successfully**", state="complete")
    
    # ── Step 3: Evaluation Quick Check ──
    with st.status("📊 **Step 3/3 — Running Quick Evaluation**", expanded=True) as status:
        st.write("Testing retrieval quality: **BM25 vs Vector vs Hybrid Fusion**")
        
        eval_query = "What is the main contribution of this paper?"
        bm25_res = bm25.search(eval_query, top_k=5)
        vec_res = vector.search(eval_query, top_k=5)
        hyb_res = reciprocal_rank_fusion(
            bm25.search(eval_query, top_k=60),
            vector.search(eval_query, top_k=60)
        )[:5]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("BM25 Hits", len(bm25_res))
        col2.metric("Vector Hits", len(vec_res))
        col3.metric("Hybrid Fused", len(hyb_res))
        
        if hyb_res:
            st.write("**Top Hybrid Result:**")
            top = hyb_res[0]
            st.caption(f"Page {top.get('page')} — {top.get('text', '')[:200]}...")
        
        # Init Gemini client
        try:
            client = GeminiMultimodalClient(model_name="gemini-3.5-flash")
            st.session_state.client = client
            st.write("✅ Gemini API connection verified")
        except Exception as e:
            st.warning(f"⚠️ Gemini API: {e}. Chat will not work without a valid API key.")
        
        status.update(label="✅ **Step 3/3 — Evaluation Complete**", state="complete")
    
    st.session_state.pipeline_stage = "ready"
    st.balloons()
    st.rerun()


# ─── Stage: Ready for Q&A ─────────────────────────────────────
if st.session_state.pipeline_stage == "ready":
    st.markdown("---")
    
    chunks = st.session_state.chunks
    bm25 = st.session_state.bm25
    vector = st.session_state.vector
    client = st.session_state.client
    
    if not chunks or not bm25 or not vector:
        st.error("Pipeline state lost. Please re-upload the PDF.")
        st.session_state.pipeline_stage = "upload"
        st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "evidence" in msg and msg["evidence"]:
                with st.expander("🔍 View Retrieved Evidence", expanded=False):
                    for i, chunk in enumerate(msg["evidence"], 1):
                        st.markdown(f"**[{i}] Page {chunk.get('page')} · {chunk.get('type', 'text').upper()} · Section: {chunk.get('section', 'N/A')}**")
                        if chunk.get("text"):
                            st.caption(chunk["text"][:400])
                        if chunk.get("image_path") and os.path.exists(chunk["image_path"]):
                            st.image(chunk["image_path"], caption=chunk.get("caption", "Extracted Image"), width=400)
                        if chunk.get("linked_images"):
                            for img in chunk["linked_images"]:
                                if os.path.exists(img):
                                    st.image(img, caption="Linked Visual Evidence", width=400)
                        st.divider()
    
    # Chat input
    if query := st.chat_input("Ask anything about the paper..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving and reasoning..."):
                # 1. Retrieve
                deep_bm25 = bm25.search(query, top_k=60)
                deep_vector = vector.search(query, top_k=60)
                
                # 2. Fuse
                fused_res = reciprocal_rank_fusion(deep_bm25, deep_vector)[:5]
                
                # 3. Link Evidence
                enriched_chunks = link_evidence(fused_res, chunks)
                
                # 4. Generate
                if client:
                    try:
                        answer = client.generate_answer(query, enriched_chunks)
                        st.markdown(answer)
                        
                        with st.expander("🔍 View Retrieved Evidence", expanded=False):
                            for i, chunk in enumerate(enriched_chunks, 1):
                                st.markdown(f"**[{i}] Page {chunk.get('page')} · {chunk.get('type', 'text').upper()} · Section: {chunk.get('section', 'N/A')}**")
                                if chunk.get("text"):
                                    st.caption(chunk["text"][:400])
                                if chunk.get("image_path") and os.path.exists(chunk["image_path"]):
                                    st.image(chunk["image_path"], caption=chunk.get("caption", "Extracted Image"), width=400)
                                if chunk.get("linked_images"):
                                    for img in chunk["linked_images"]:
                                        if os.path.exists(img):
                                            st.image(img, caption="Linked Visual Evidence", width=400)
                                st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "evidence": enriched_chunks,
                        })
                    except Exception as e:
                        st.error(f"API Error: {e}")
                else:
                    st.error("Gemini API client not initialized. Please check your API key in `.env`.")

