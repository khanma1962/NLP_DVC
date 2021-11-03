
import os
import logging
import yaml

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f'yaml file {path_to_yaml} has been successfully read')

    return content


