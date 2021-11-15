import numpy as np
import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, get_df
import joblib
from sklearn.ensemble import RandomForestClassifier

STAGE = 'three'

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def main(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    # print(f"featurized_data_dir_path is {featurized_data_dir_path}")
    featurized_train_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TRAIN"])
    
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"] )
    create_directories([model_dir_path])
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    matrix = joblib.load(featurized_train_data_path)
    labels = np.squeeze(matrix[:, 1].toarray())  # all rows and first col. we need a list not an array
    X      = matrix[:, 2:] # second col is data

    logging.info(f"input matrix size is {matrix.shape}")
    logging.info(f"X matrix size is {X.shape}")
    logging.info(f"labels  size is {labels.shape}")

    seed   = params["train"]["seed"]
    n_est  = params["train"]["n_est"]
    min_split= params["train"]["min_split"]

    model = RandomForestClassifier(
        n_estimators=n_est,
        min_samples_split= min_split,
        n_jobs=2,
        random_state=  seed
        )

    model.fit(X, labels)

    joblib.dump(model, model_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed! all the data are saved in local <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e