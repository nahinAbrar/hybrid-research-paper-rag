from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

def show_first_five_chunks(chunks): 
    print("First 5 chunks of the document:")
    for i, chunk in enumerate(chunks[5:10]):
        print(f"Chunk {i + 1}:")
        print(chunk.page_content)
        print("-" * 40)

def main():
    print("Starting the ingestion pipeline...")

    pdf_dir = Path(r"E:\Study materials\CSE713")

    if not pdf_dir.exists():
        print(f"Path not found: {pdf_dir}")
        return

    if pdf_dir.is_file():
        docs = PyMuPDFLoader(str(pdf_dir)).load()
    else:
        loader = DirectoryLoader(str(pdf_dir), glob="*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()

    if not docs:
        print(f"No PDF files found in {pdf_dir}")
        return

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Split document into {len(chunks)} chunks.")

    show_first_five_chunks(chunks)

#Write a method for showing me first 5 chunks of the document

    

if __name__ == "__main__":
    main()