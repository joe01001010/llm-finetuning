"""
Full training pipeline for Qwen 7B fine-tuning.

This script handles:
- Pulling data from Kaggle
- Data preprocessing
- LoRA fine-tuning
- PPO optimization
- Evaluation and logging

Author: Joe Weibel
Date: 26 Feb 2026
"""
#!/usr/bin/env python


from pathlib import Path
import kagglehub


def pull_data(dataset_name, output_directory):
    """
    This function takes two arguments
    One argument is the output directory to download the dataset to
    This function takes the path for a dataset as an argument
    This function will pull the dataset from kaggle and download it locally
    This function will return the path to the dataset
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    return kagglehub.dataset_download(dataset_name,
        output_dir=str(output_directory),
        force_download=True,)


def main():
    """
    This program is designed to pull data and save it locally from the internet
    This program will clean and format the data for the AI agent to receive
    """
    weather_data_path = pull_data("ananthr1/weather-prediction",
        Path(__file__).resolve().parent.parent / "data" / "weather-prediction")
    print(weather_data_path)


if __name__ == '__main__':
    main()
