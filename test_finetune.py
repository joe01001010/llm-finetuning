#!/usr/bin/env python
# test_finetune.py - Quick test of the fine-tuning pipeline
import subprocess
import sys
import os

def run_command(command):
  """Run shell command and print output"""
  print(f"\n>>> Running: {command}")
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  if result.stdout:
    print("Output:", result.stdout)
  if result.stderr:
    print("Errors:", result.stderr)
  return result.returncode

def main():
  print("=== Testing Fine-tuning Pipeline ===")
  
  # Step 1: Create virtual environment
  if not os.path.exists("venv"):
      print("\n1. Creating virtual environment...")
      run_command("python3 -m venv venv")
  
  # Step 2: Activate and install packages
  print("\n2. Installing dependencies...")
  
  # Install in one go
  install_cmd = (
      "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && "
      "pip install transformers datasets accelerate peft bitsandbytes && "
      "pip install sentencepiece protobuf scipy pandas faker"
  )
  
  if sys.platform == "win32" or "WSL" in sys.platform:
    # For WSL/Windows
    run_command(f"source venv/bin/activate && {install_cmd}")
  else:
    run_command(f". venv/bin/activate && {install_cmd}")
  
  # Step 3: Create dummy data
  print("\n3. Creating dummy dataset...")
  run_command("python create_fake_data.py")
  
  # Step 4: Test with a smaller model first
  print("\n4. Testing with tiny model...")
  
  # Create a minimal test script
  test_script = """
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Test with a tiny model
model_name = "gpt2"  # Very small for quick test
print(f"Testing with {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("✓ Model loaded successfully")

# Test generation
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
with torch.no_grad():
  outputs = model.generate(**inputs, max_length=50)
print("✓ Generation test passed")
print("Output:", tokenizer.decode(outputs[0]))
"""
  
  with open("quick_test.py", "w") as f:
    f.write(test_script)
  
  run_command("python quick_test.py")
  
  print("\n=== Pipeline Test Complete ===")
  print("\nNext steps:")
  print("1. Review the dummy data: cat dummy_training_data.json | head -5")
  print("2. Run full fine-tuning: python finetune.py")
  print("3. Test inference: python inference.py")
  
  # Cleanup
  if os.path.exists("quick_test.py"):
    os.remove("quick_test.py")

if __name__ == '__main__':
  main()
