import numpy as np
import argparse
import os

import math
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, save_json
import joblib
import sklearn.metrics as metrics

STAGE = 'four'

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def main(config_path):
    config = read_yaml(config_path)

    artifacts = config["artifacts"]

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    # print(f"featurized_data_dir_path is {featurized_data_dir_path}")

    featurized_test_data_path  = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TEST"])
    
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"] )
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    model = joblib.load(model_path)
    matrix = joblib.load(featurized_test_data_path)
    labels = np.squeeze(matrix[:, 1].toarray())  # all rows and first col. we need a list not an array
    X      = matrix[:, 2:] # second col is data

    predictions_by_class = model.predict_proba(X)
    predcitions = predictions_by_class[:, 1]

    PRC_json_path = config["plots"]["PRC"]
    ROC_json_path = config["plots"]["ROC"]

    score_json_path = config["metrics"]["SCORES"]

    precision, recall, prc_threshold = metrics.precision_recall_curve(labels, predcitions)
    fpr, tpr, roc_threshold = metrics.roc_curve(labels, predcitions)

    avg_prec = metrics.average_precision_score(labels, predcitions)

    roc_auc_score = metrics.roc_auc_score(labels, predcitions)

    scores = {
        "avg_prec" : avg_prec,
        "roc_auc" : roc_auc_score,
        }

    save_json(score_json_path, scores)

    precision, recall, prc_threshold = metrics.precision_recall_curve(labels, predcitions)

    nth_point = math.ceil(len(prc_threshold) / 1000) # rounding the number to the highest point
    
    prc_points = list(zip(precision, recall, prc_threshold))[: :nth_point]

    prc_data = {
        "prc" : [{"precision": p, "recall": r, "threshold": t} for p, r, t in prc_points
        ]
        }

    save_json(PRC_json_path, prc_data)

    fpr, tpr, roc_threshold = metrics.roc_curve(labels, predcitions)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed! all the data are saved in local <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e