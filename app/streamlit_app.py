"""
Streamlit web application for the interactive RAG demo.
"""
import streamlit as st

def main():
    """Main Streamlit app execution."""
    st.title("Hybrid Multimodal RAG for Research Papers")
    st.write("Upload a research paper PDF and ask questions about its text, tables, and figures.")

    # TODO: implement
    # 1. File uploader for PDF
    # 2. Trigger parsing pipeline
    # 3. Trigger indexing (BM25 + Vector)
    # 4. Chat interface for Q&A
    # 5. Display answer and retrieved evidence (text snippets + images)

if __name__ == "__main__":
    main()
