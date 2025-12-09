# scripts/1_build_index.py

import os
# LangChain components for data handling and ChromaDB
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # TextLoader is crucial here
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/knowledge"
PERSIST_DIR = "vector_store/chroma_db"
# A lightweight model good for RAG on smaller devices
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def build_index():
    """Loads documents, splits them, creates embeddings, and saves the Chroma index."""
    print("--- ATLAS Indexing Started ---")

    # 1. Load Documents from the knowledge folder (FIX APPLIED HERE)
    try:
        # Load .txt and .md files recursively, forcing the use of TextLoader
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.txt",  # Explicitly search for .txt files
            loader_cls=TextLoader,  # Tell it to use TextLoader for all matched files
            recursive=True,
            show_progress=True
        )
        documents = loader.load()

        # NOTE: If you also have .md files, you would typically add a second loader for them
        # or combine the loading. For now, we focus on the .txt files provided.

        if not documents:
            print(f"ERROR: No documents found in {DATA_PATH} matching the glob pattern '**/*.txt'.")
            print("Please ensure your files are present and have the .txt extension.")
            return
    except Exception as e:
        print(f"An error occurred during document loading: {e}")
        return

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\n- Split {len(documents)} documents into {len(chunks)} chunks.")

    # 3. Create the Embedding Function
    print(f"- Using lightweight SentenceTransformer: {EMBEDDING_MODEL_NAME}")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Ensure the persistence directory exists before saving
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # 4. Create and Persist the Chroma Index
    # This is the main RAG component creation step
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    db.persist()

    print(f"--- Indexing Complete. Index saved to: {PERSIST_DIR} ---")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    build_index()