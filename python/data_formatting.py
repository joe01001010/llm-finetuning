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
import json
import kagglehub
import pandas as pd


SYSTEM = "You are a weather forecasting assistant. You should only output in JSON format"
FEATURE_COLUMNS = ['temp_max', 'temp_min', 'precipitation', 'wind', 'weather']
EXTERNAL_DATASET = "seattle-weather.csv"
WINDOW_SIZES = [7, 14, 21, 28]
JSON_OUT_PATH = "seattle_weather_chat.json"


def format_day(row, day_offset):
    """
    This function takes two arguments
    Row should be a dictionary of the day's representation
    day_offset should be a negative number counting backwards
    This function will return a tuple of strings
    """
    return (
      f"Day {day_offset} ({row['date'].date()}): "
      f"temp_max={row['temp_max']}, temp_min={row['temp_min']}, "
      f"precipitation={row['precipitation']}, wind={row['wind']}, "
      f"weather={row['weather']}"
    )


def format_target(row):
    """
    This function takes one argument
    row is the row to disect into a string and set as the target
    This function will return a dictionary of the weather keys and values as json
    """
    target = {
        "temp_max": float(row['temp_max']),
        "temp_min": float(row['temp_min']),
        "precipitation": float(row['precipitation']),
        "wind": float(row['wind']),
        "weather": row['weather'],
    }
    return json.dumps(target, separators=(",", ":"))


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


def format_data(path_to_dataset):
    """
    This function takes one argument
    path_to_dataset is a path to the csv to import the data from
    This function will sort the data by date and delete duplicates and na columns
    This function will return a pandas dataframe of the external data
    """
    df = pd.read_csv(path_to_dataset)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


def write_samples(data_frame):
    """
    This function takes one argument as the dataframe
    data_frame is the pandas data structure holding the external data
    This function will format each sample into a sample the LLM can understand
    This function will write the samples to a json file
    This function doesnt return anything
    """
    sample_count = 0
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as file:
        for w in WINDOW_SIZES:
            for t in range(w, len(data_frame)):
                window = data_frame.iloc[t - w:t]
                target_row = data_frame.iloc[t]

                lines = [format_day(window.iloc[i], i - w) for i in range(w)]
                user = (
                    f"Given the last {w} days of Seattle weather, predict tomorrow. \n"
                    + "\n".join(lines)
                    + "\nReturn JSON with keys: temp_max, temp_min, precipitation, wind, weather."
                )

                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": format_target(target_row)},
                    ]
                }
                sample_count += 1
                file.write(json.dumps(example) + "\n")

    print(f"Finished writing {sample_count} samples to {JSON_OUT_PATH}")


def main():
    """
    This program is designed to pull data and save it locally from the internet
    This program will clean and format the data for the AI agent to receive
    """
    weather_data_path = pull_data("ananthr1/weather-prediction",
        Path(__file__).resolve().parent.parent / "data" / "weather-prediction")
    pandas_data = format_data(weather_data_path + "/" + EXTERNAL_DATASET)
    write_samples(pandas_data)



if __name__ == '__main__':
    main()
