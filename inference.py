#!/usr/bin/env python
# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch

def test_base_model():
  """Test the base model before fine-tuning"""
  print("Testing base model...")
  
  # Use a smaller model for quick testing
  model_name = "Qwen/Qwen2.5-0.5B-Instruct"
  
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch.float16,
      device_map="auto",
      trust_remote_code=True
  )
  
  # Test prompt
  test_prompts = [
      "### Instruction: What is Quantum AI Platform?\n### Response:",
      "### Instruction: Explain cloud computing\n### Response:",
  ]
  
  for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=100,
          temperature=0.7,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id
      )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}\n")

def test_fine_tuned_model(model_path="./fine_tuned_model"):
  """Test the fine-tuned model"""
  try:
    print("\nTesting fine-tuned model...")
    
    # Load base config
    config = PeftConfig.from_pretrained(model_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test with dummy data prompts
    test_prompts = [
        "### Instruction: What is NeuralSync Pro?\n### Response:",
        "### Instruction: How does DataFlow Analytics Suite benefit users?\n### Response:",
        "### Instruction: What are the features of CloudSecure Enterprise?\n### Response:",
    ]
    
    for prompt in test_prompts:
      print(f"\n{'='*50}")
      print(f"Prompt: {prompt}")
      
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      
      with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
      
      response = tokenizer.decode(outputs[0], skip_special_tokens=True)
      
      # Extract just the response part
      if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
      
      print(f"Response: {response}")
    
    print(f"\n{'='*50}")
    print("✓ Fine-tuned model test complete!")
      
  except Exception as e:
    print(f"Could not load fine-tuned model: {e}")
    print("You may need to run finetune.py first")

def interactive_chat():
  """Interactive chat with the model"""
  try:
    # Try to load fine-tuned model first
    config = PeftConfig.from_pretrained("./fine_tuned_model")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, "./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    print("Loaded fine-tuned model for interactive chat")
  except:
    # Fall back to base model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded base model for interactive chat")
  
  print("\n=== Interactive Chat ===")
  print("Type 'quit' to exit")
  print("Format your questions as: What is [product]? or Explain [concept]")
  
  while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
      break
    
    # Format prompt
    prompt = f"### Instruction: {user_input}\n### Response:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=200,
          temperature=0.7,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id
      )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in response:
      response = response.split("### Response:")[-1].strip()
    
    print(f"\nAssistant: {response}")

if __name__ == '__main__':
  print("=== Model Inference Test ===")
  
  # Test 1: Base model
  test_base_model()
  
  # Test 2: Fine-tuned model (if exists)
  test_fine_tuned_model()
  
  # Test 3: Interactive chat
  # Uncomment to try interactive mode
  # interactive_chat()
