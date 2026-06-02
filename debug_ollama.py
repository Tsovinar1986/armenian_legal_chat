#!/usr/bin/env python3
"""
Quick diagnostic script to check Ollama setup
"""
import sys
import os
import subprocess

print("=" * 60)
print("OLLAMA DIAGNOSTIC CHECK")
print("=" * 60)

# Check 1: Is Ollama running?
print("\n1️⃣ Checking if Ollama service is running...")
try:
    result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                          capture_output=True, text=True, timeout=3)
    if result.returncode == 0:
        print("✅ Ollama service is running")
        print(f"   Response: {result.stdout[:200]}")
    else:
        print("❌ Ollama service is NOT responding")
        print("   Run: ollama serve")
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    print("   Make sure Ollama is running: ollama serve")

# Check 2: Try to import and use langchain_ollama
print("\n2️⃣ Checking langchain_ollama availability...")
try:
    from langchain_ollama import OllamaLLM
    print("✅ langchain_ollama imported successfully")
    
    # Check 3: Try to initialize the model
    print("\n3️⃣ Attempting to initialize armenian-lawyer-router model...")
    llm = OllamaLLM(model="armenian-lawyer-router")
    print("✅ Model initialized successfully")
    
    # Check 4: Try to invoke the model with a test prompt
    print("\n4️⃣ Testing model with a simple prompt...")
    test_prompt = "Ինչ-որ շատ կարճ թեստ: Պատասխան տուր մեկ հատ նախադասենում:"
    response = llm.invoke(test_prompt)
    print(f"✅ Model responded successfully!")
    print(f"   Response: {response[:200]}")
    
except ImportError as e:
    print(f"❌ langchain_ollama not installed: {e}")
    print("   Run: pip install langchain-ollama")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    print(f"   Model name: armenian-lawyer-router")
    print("   Available commands:")
    print("   - Check available models: ollama list")
    print("   - Pull model: ollama pull armenian-lawyer-router")
    print("   - List all Ollama models: ollama list")

print("\n" + "=" * 60)
print("END OF DIAGNOSTIC CHECK")
print("=" * 60)
