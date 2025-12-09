cat > ~/atlas_project/test_llm.py << 'EOF'
import sys
import traceback

print("🔍 Starting LLM load test...")
print(f"Python version: {sys.version}")

try:
    print("\n1️⃣ Importing llama_cpp...")
    from llama_cpp import Llama
    print("✅ Import successful")
    
    print("\n2️⃣ Attempting to load model...")
    llm = Llama(
        model_path="models/gemma-2b-q4_k_m.gguf",
        n_ctx=2048,
        n_threads=4,
        n_batch=512,
        verbose=True
    )
    print("✅ Model loaded successfully!")
    
    print("\n3️⃣ Testing inference...")
    response = llm("Hello", max_tokens=10)
    print(f"✅ Response: {response}")
    
except Exception as e:
    print(f"\n❌ ERROR OCCURRED:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\n📋 Full traceback:")
    traceback.print_exc()
    
    # Try to get more info
    import os
    print(f"\n📁 Model file exists: {os.path.exists('models/gemma-2b-q4_k_m.gguf')}")
    if os.path.exists('models/gemma-2b-q4_k_m.gguf'):
        print(f"📊 Model file size: {os.path.getsize('models/gemma-2b-q4_k_m.gguf') / (1024**3):.2f} GB")

print("\n🏁 Test completed")
EOF
