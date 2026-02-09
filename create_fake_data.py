#!/usr/bin/env python
# create_fake_data.py
import pandas as pd
import json
import random
from faker import Faker

def create_dummy_dataset(num_samples=100):
    """Create a dummy dataset for testing fine-tuning"""
    
    fake = Faker()
    
    # Sample instruction templates
    instruction_templates = [
        "Explain what {product} does",
        "What are the main features of {product}?",
        "How does {product} benefit users?",
        "What problem does {product} solve?",
        "Describe the use cases for {product}",
        "Who is the target audience for {product}?",
        "What makes {product} unique?",
        "How does {product} compare to alternatives?",
        "What are the technical specifications of {product}?",
        "What industries use {product}?"
    ]
    
    # Sample products
    products = [
        "Quantum AI Platform",
        "NeuralSync Pro",
        "DataFlow Analytics Suite",
        "CloudSecure Enterprise",
        "IoT Edge Processor",
        "BlockChain Verifier",
        "AutoML Studio",
        "CyberShield Firewall",
        "BioMetrics Scanner",
        "SmartContract Auditor"
    ]
    
    # Sample features/benefits
    features = [
        "real-time data processing",
        "machine learning capabilities",
        "blockchain integration",
        "advanced security protocols",
        "scalable infrastructure",
        "user-friendly interface",
        "customizable workflows",
        "cross-platform compatibility",
        "automated reporting",
        "predictive analytics"
    ]
    
    data = []
    
    for i in range(num_samples):
      product = random.choice(products)
      instruction_template = random.choice(instruction_templates)
      
      # Create instruction with product name
      instruction = instruction_template.format(product=product)
      
      # Create input (product description)
      input_text = f"{product} is a cutting-edge solution that offers {random.choice(features)}. "
      input_text += f"It was developed by {fake.company()} and launched in {fake.year()}. "
      input_text += f"The platform supports {random.randint(1, 10)} programming languages and has {random.randint(100, 10000)} active users."
        
      # Create output (response)
      output = f"{product} is designed to {instruction_template.replace('{product}', '').lower().strip()}. "
      output += f"It provides {random.choice(features)} and {random.choice(features)}. "
      output += f"Key benefits include: "
      output += f"1. {fake.sentence()} "
      output += f"2. {fake.sentence()} "
      output += f"3. {fake.sentence()} "
      output += f"Typical users are {fake.job()}s and {fake.job()}s in the {fake.bs()} industry."
      
      data.append({
          "id": f"sample_{i+1:03d}",
          "instruction": instruction,
          "input": input_text,
          "output": output,
          "product": product,
          "category": random.choice(["AI", "Security", "Analytics", "IoT", "Blockchain"])
      })
    
    return data

def save_datasets(data, formats=['json', 'csv']):
  """Save dataset in multiple formats"""
  
  df = pd.DataFrame(data)
  
  if 'json' in formats:
    # Save as JSON
    with open('dummy_training_data.json', 'w') as f:
      json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to dummy_training_data.json")
  
  if 'csv' in formats:
    # Save as CSV
    df.to_csv('dummy_training_data.csv', index=False)
    print(f"Saved {len(data)} samples to dummy_training_data.csv")
  
  # Also save a smaller test dataset
  test_data = data[:5]
  with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)
  print(f"Saved {len(test_data)} test samples to test_data.json")
  
  return df

def create_q_format_data(data):
  """Create Q-formatted data (Qwen specific format)"""
  q_data = []
  for item in data:
    q_data.append({
        "conversations": [
            {
                "role": "user",
                "content": f"{item['instruction']}\n{item['input']}"
            },
            {
                "role": "assistant",
                "content": item['output']
            }
        ]
    })
  
  with open('dummy_q_format_data.json', 'w') as f:
    json.dump(q_data, f, indent=2)
  print(f"Saved Q-formatted data to dummy_q_format_data.json")
  
  return q_data

if __name__ == '__main__':
  # Install faker if not installed
  try:
    from faker import Faker
  except ImportError:
    print("Installing Faker...")
    import subprocess
    subprocess.check_call(["pip", "install", "Faker"])
    from faker import Faker
  
  # Create 50 dummy samples (good for testing)
  print("Creating dummy dataset...")
  dummy_data = create_dummy_dataset(num_samples=50)
  
  # Save in multiple formats
  df = save_datasets(dummy_data, formats=['json', 'csv'])
  
  # Create Q-formatted data
  q_data = create_q_format_data(dummy_data[:10])
  
  # Print sample
  print("\nSample data point:")
  print(json.dumps(dummy_data[0], indent=2))
  
  # Print statistics
  print(f"\nDataset Statistics:")
  print(f"Total samples: {len(dummy_data)}")
  print(f"Unique products: {df['product'].nunique()}")
  print(f"Unique categories: {df['category'].nunique()}")
