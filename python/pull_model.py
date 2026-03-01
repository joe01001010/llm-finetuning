"""
pull_model.py
This program is designed to pull QWEN 7B
"""
#!/usr/bin/env python


import os
from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2-7B-Instruct"
MODEL_DIR = f"/local-containers/{MODEL_ID.rsplit('/', maxsplit=1)[-1]}"


def main():
    """
    This function will check if the model dir exists locally
    If the model directory doesnt exist it will download the model
    """
    if os.path.exists(MODEL_DIR):
        print(f"Model {MODEL_ID} is already downloaded")
    else:
        path = snapshot_download(
            repo_id = MODEL_ID,
            local_dir = MODEL_DIR,
            local_dir_use_symlinks = False,
        )
        print(f"Downloaded model {MODEL_ID}: {path}")


if __name__ == '__main__':
  main()
