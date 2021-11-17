
import pandas as pd
import os
import logging
import yaml
import json

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f'yaml file {path_to_yaml} has been successfully read')

    return content

def create_directories(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"dirctory created as {dir_path}")

def get_df(path_to_data: str, sep: str="\t") -> pd.DataFrame:
    df = pd.read_csv(
        path_to_data,
        encoding="utf-8",
        header=None,
        delimiter="\t",
        names=["id", "label", "text"]
        )
    logging.info(f"DataFrame has been created at {path_to_data} of size {df.shape}")
    
    return df


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"Json file is stored and {path}")




