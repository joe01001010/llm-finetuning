# prepare_data.py
import pandas as pd
import json


def main():
  # Example: Convert your proprietary data to training format
  # Format 1: Instruction-following format
  data = [
      {
          "instruction": "What is this product used for?",
          "input": "Product XYZ specifications...",
          "output": "Product XYZ is primarily used for..."
      },
      # Add more examples
  ]
  
  # Save as JSON
  with open('training_data.json', 'w') as f:
      json.dump(data, f, indent=2)
  
  # Or save as CSV
  df = pd.DataFrame(data)
  df.to_csv('training_data.csv', index=False)

if __name__ == '__main__':
  main()
