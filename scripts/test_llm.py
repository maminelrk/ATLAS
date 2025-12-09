# scripts/test_llm.py

import os
from llama_cpp import Llama

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the exact file name of your downloaded GGUF model
MODEL_FILE_NAME = "gemma-2b-q4_k_m.gguf"
MODEL_PATH = os.path.join("model_files", MODEL_FILE_NAME)

# --- INFERENCE ---
try:
    print(f"Loading model from: {MODEL_PATH}...")

    # 1. Initialize the Llama model object
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,  # Context length (can be adjusted)
        n_gpu_layers=0,  # Set to 0 to force CPU usage for testing
        verbose=True
    )

    # 2. Define the prompt in an instruction format
    prompt = "Write a short, fun fact about Morocco."

    print("-" * 50)
    print(f"User Prompt: {prompt}")

    # 3. Generate the response
    output = llm(
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        max_tokens=128,
        stop=["<end_of_turn>"],
        echo=False
    )

    # 4. Extract and print the text
    response_text = output['choices'][0]['text'].strip()
    print("\nLLM Response:")
    print(response_text)
    print("-" * 50)

except FileNotFoundError:
    print(f"\nERROR: Model file not found at {MODEL_PATH}.")
    print("Please check that the file name is correct and the file is in the 'model_files' directory.")
except Exception as e:
    print(f"\nAn error occurred during model inference: {e}")