import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]

def load_and_split_documents(file_paths, chunk_size=1000, chunk_overlap=200):
    """
    Load multiple documents (PDF, DOCX, TXT), extract text and split into chunks.

    Args:
        file_paths (list): List of file paths to process.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: List of LangChain Document chunks.
    """
    all_docs = []
    for path in file_paths:
        ext = path.split(".")[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file: {path}")
            continue

        if ext == "pdf":
            print("Entered PDF loader")
            loader = PyPDFLoader(path)
        elif ext == "docx":
            loader = Docx2txtLoader(path)
        elif ext == "txt":
            loader = TextLoader(path, encoding="utf-8")

        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(all_docs)
    # chunks = text_splitter.split_text(all_docs)
    print(len(chunks))
    print(chunks[:5])
    print(f"Loaded {len(all_docs)} documents, created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    chunks = load_and_split_documents("D:/GenAI-Practice/GenAI-Projects/offline_rag/Attention Is All You Need.pdf")
    print(len(chunks))
    print(chunks[:5])