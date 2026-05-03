"""
Dataset preparation for the Seattle weather fine-tuning project.

This script:
- pulls the raw Kaggle dataset
- cleans and sorts the weather records
- formats chat-style SFT examples
- writes train and eval JSONL files that can also be converted into
  prompt-only PPO rollouts later
"""
#!/usr/bin/env python


from pathlib import Path
import json
import pandas as pd


SYSTEM = "You are a weather forecasting assistant. You should only output in JSON format"
TRAIN_SPLIT = 0.8
FEATURE_COLUMNS = ['temp_max', 'temp_min', 'precipitation', 'wind', 'weather']
EXTERNAL_DATASET = "seattle-weather.csv"
WINDOW_SIZES = [7, 14, 21, 28]
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / "data"
LEGACY_JSON_OUT_PATH = DATA_DIR / "seattle_weather_chat.json"
JSONL_OUT_PATH = DATA_DIR / "seattle_weather_chat.jsonl"
TRAIN_OUT_PATH = DATA_DIR / "seattle_weather_chat_train.jsonl"
EVAL_OUT_PATH = DATA_DIR / "seattle_weather_chat_eval.jsonl"


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
    import kagglehub

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


def build_window_samples(data_frame, window_size):
    """
    This function takes the cleaned dataframe and one window size
    This function will build every augmented sample for that window size
    This function returns a list of examples in chronological order
    """
    samples = []
    for t in range(window_size, len(data_frame)):
        window = data_frame.iloc[t - window_size:t]
        target_row = data_frame.iloc[t]

        lines = [format_day(window.iloc[i], i - window_size) for i in range(window_size)]
        user = (
            f"Given the last {window_size} days of Seattle weather, predict tomorrow. \n"
            + "\n".join(lines)
            + "\nReturn JSON with keys: temp_max, temp_min, precipitation, wind, weather."
        )

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": format_target(target_row)},
            ]
        })
    return samples


def split_samples(samples):
    """
    This function takes the full augmented sample list for one window size
    This function will keep the earliest 80 percent for training
    This function will keep the final 20 percent for evaluation
    """
    if len(samples) < 2:
        return samples, []

    split_index = int(len(samples) * TRAIN_SPLIT)
    split_index = min(max(split_index, 1), len(samples) - 1)
    return samples[:split_index], samples[split_index:]


def write_jsonl(path, samples):
    """
    This function takes a file path and a sample list
    This function writes newline-delimited json for downstream scripts
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample) + "\n")


def write_samples(data_frame):
    """
    This function takes one argument as the dataframe
    data_frame is the pandas data structure holding the external data
    This function will format each sample into a sample the LLM can understand
    This function will write combined, training, and evaluation jsonl files
    This function doesnt return anything
    """
    all_samples = []
    train_samples = []
    eval_samples = []

    for window_size in WINDOW_SIZES:
        window_samples = build_window_samples(data_frame, window_size)
        split_train, split_eval = split_samples(window_samples)

        all_samples.extend(window_samples)
        train_samples.extend(split_train)
        eval_samples.extend(split_eval)

    write_jsonl(JSONL_OUT_PATH, all_samples)
    write_jsonl(LEGACY_JSON_OUT_PATH, all_samples)
    write_jsonl(TRAIN_OUT_PATH, train_samples)
    write_jsonl(EVAL_OUT_PATH, eval_samples)

    print(f"Finished writing {len(all_samples)} total samples to {JSONL_OUT_PATH}")
    print(f"Updated legacy combined dataset at {LEGACY_JSON_OUT_PATH}")
    print(f"Finished writing {len(train_samples)} training samples to {TRAIN_OUT_PATH}")
    print(f"Finished writing {len(eval_samples)} evaluation samples to {EVAL_OUT_PATH}")


def main():
    """
    This program is designed to pull data and save it locally from the internet
    This program will clean and format the data for the AI agent to receive
    """
    weather_data_path = pull_data("ananthr1/weather-prediction",
        PROJ_DIR / "data" / "weather-prediction")
    pandas_data = format_data(Path(weather_data_path) / EXTERNAL_DATASET)
    write_samples(pandas_data)



if __name__ == '__main__':
    main()
