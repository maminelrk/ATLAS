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
MODEL_FILE_NAME = "gemma-2b-q4_k_m.gguf"

# FIX: Build path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to ATLAS/
MODEL_PATH = os.path.join(project_root, "model_files", MODEL_FILE_NAME)

# --- PROMPT TEMPLATE ---
# ... (rest unchanged) ...

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
    print(f"Loading model from: {MODEL_PATH}...")
    
    # Verify file exists before loading
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print(f"   Please verify the file exists at this location.")
        return
    
    print(f"✅ Model file found. Size: {os.path.getsize(MODEL_PATH) / (1024**3):.2f} GB")
    
    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=0,      # CPU only
            n_ctx=2048,          # Reduced for Pi 5 (was 4096)
            n_threads=4,         # Optimize for Pi 5
            n_batch=512,         # Optimize for Pi 5
            temperature=0.1,
            verbose=True,        # Set to True to see loading progress
            stop=["<end_of_turn>"],
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR loading LlamaCpp model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Create the Prompt and Retrieval Chain
    # ... (rest of your code) ...
