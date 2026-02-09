#!/usr/bin/env bash
# run_all.sh - Complete setup and test script

echo "=== Setting up LLM Fine-tuning Project ==="

# Step 1: Create and activate virtual environment
echo "1. Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 2: Install dependencies
echo "2. Installing dependencies..."
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers datasets accelerate peft bitsandbytes
pip install sentencepiece protobuf scipy pandas faker

# Step 3: Create dummy data
echo "3. Creating dummy dataset..."
python create_fake_data.py

# Step 4: Show sample data
echo "4. Sample data created:"
echo "--- First training sample ---"
head -50 dummy_training_data.json | tail -20
echo "---"

# Step 5: Test download (optional - comment if slow)
# echo "5. Testing model download..."
# python download_model.py

# Step 6: Run quick test
echo "6. Running pipeline test..."
python test_finetune.py

echo "=== Setup Complete ==="
echo ""
echo "To run fine-tuning:"
echo "  source venv/bin/activate"
echo "  python finetune.py"
echo ""
echo "To test inference:"
echo "  python inference.py"
echo ""
echo "To create more dummy data:"
echo "  python create_fake_data.py"
