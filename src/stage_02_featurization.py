import numpy as np
import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, get_df
from src.utils.featurize import save_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

STAGE = 'two'

def main(config_path, params_path):
    
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    prepare_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    train_data_path = os.path.join(prepare_data_dir_path, artifacts["TRAIN_DATA"])
    test_data_path  = os.path.join(prepare_data_dir_path, artifacts["TEST_DATA"])

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    # print(f"featurized_data_dir_path is {featurized_data_dir_path}")
    create_directories([featurized_data_dir_path])
    featurized_train_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TRAIN"])
    featurized_test_data_path  = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TEST"])

    max_features = params["featurize"]["max_feature"]
    ngrams = params["featurize"]["ngrams"]
    
    df_train = get_df(train_data_path)

    train_words = np.array(df_train.text.str.lower().values.astype("U"))
    
    bag_of_words = CountVectorizer(
        stop_words = 'english',
        max_features = max_features,
        ngram_range= (1, ngrams)
        )
    bag_of_words.fit(train_words)
    
    train_words_binary_matrix = bag_of_words.transform(train_words)
    # print(f"bag of words are {train_words_binary_matrix.toarray()}")

    tfidf = TfidfTransformer(smooth_idf= False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)
    save_matrix(df_train, train_words_binary_matrix, featurized_train_data_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE}  completed! Featurization done !!! <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e

