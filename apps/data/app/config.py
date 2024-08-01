import os
from typing import Dict
from pydantic import BaseSettings
import json
import pandas as pd
import numpy as np
import os

class Settings(BaseSettings):
    """Settings read from the .env file in the parent directory"""
    assets_path: str
    data_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()


datasets = {}

dataset_cfgs = [os.path.join(settings.assets_path,i) for i in os.listdir(settings.assets_path) if i.endswith(".json")]
for path in dataset_cfgs:
    with open(path, "r") as f:
        cfg = json.load(f)
        name = cfg['name']
        print(f"######## Load Dataset: {name} ##############")
        if cfg["dataset"]["type"] == "folder":
            if cfg["dataset"]["format"] == "csv":
                df = pd.read_csv(os.path.join(settings.assets_path,cfg["dataset"]["path"]))
        elif cfg["dataset"]["type"] == "array":
            if cfg["dataset"]["format"] == "npy":
                df = np.load(os.path.join(settings.assets_path,cfg["dataset"]["path"]))
        print(df.head())
        datasets[name] = {"cfg":cfg["dataset"], "data": df}

print(f"######## Datasets loaded ##############")
print("datasets: ",datasets)
