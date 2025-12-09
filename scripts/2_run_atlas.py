# scripts/2_run_atlas.py

import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# --- CONFIGURATION ---
PERSIST_DIR = "vector_store/chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# UPDATE THIS: Use the exact filename of your Gemma 2B GGUF model
MODEL_FILE_NAME = "gemma-2b-q4_k_m.gguf"
MODEL_PATH = os.path.join("model_files", MODEL_FILE_NAME)


# --- PROMPT TEMPLATE ---
# ... (unchanged) ...


def run_rag_test(question):
    """Initializes RAG components and answers a question."""

    # 1. Load the Embedding Model
    print("1. Loading Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 2. Load the Vector Store (ChromaDB)
    print(f"2. Loading Chroma Index from {PERSIST_DIR}...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    # 3. Initialize the LLM (Gemma 2B via LlamaCpp)
    print(f"3. Loading LLM from {MODEL_PATH}. This may take a moment...")
    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=0,  # Force CPU usage on Windows PC for consistent testing
            n_ctx=4096,  # Context window size
            temperature=0.1,  # Low temperature for factual answers
            verbose=False,
            # Adjust these for the Gemma chat template
            messages_as_prompt=True,
            stop=["<end_of_turn>"],

            
        )
    except Exception as e:
        print(f"ERROR loading LlamaCpp model: {e}")
        print("Please ensure the GGUF file is correctly named and located in the 'model_files' directory.")
        return

    # 4. Create the Prompt and Retrieval Chain
    # ... (unchanged) ...