from nltk.corpus import stopwords
import string, re
from collections import Counter
import wordcloud

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os

""" 
Text Classification with Embedding + Conv1D
by:https://www.kaggle.com/serkanpeldek/text-classification-with-embedding-conv1d
"""

review_dataset_path = "../data/review_polarity/txt_sentoken"
# print(os.listdir(review_dataset_path))

# Positive and negative reviews folder paths
pos_review_folder_path = review_dataset_path + "/" + "pos"
neg_review_folder_path = review_dataset_path + "/" + "neg"


def load_text_from_textfile(path):
    file = open(path, "r")
    review = file.read()
    file.close()
    return review


def load_review_from_textfile(path):
    return load_text_from_textfile(path)


def get_data_target(folder_path, file_names, review_type):
    data = list()
    target = list()
    for file_name in file_names:
        full_path = folder_path + "/" + file_name
        review = load_review_from_textfile(path=full_path)
        data.append(review)
        target.append(review_type)
    return data, target


if __name__ == "__main__":

    # Positive and negative file names
    pos_review_file_names = os.listdir(pos_review_folder_path)
    neg_review_file_names = os.listdir(neg_review_folder_path)

    pos_data, pos_target = get_data_target(folder_path=pos_review_folder_path,
                                           file_names=pos_review_file_names,
                                           review_type="positive")
    neg_data, neg_target = get_data_target(folder_path=neg_review_folder_path,
                                           file_names=neg_review_file_names,
                                           review_type="negative")

    print("Positive data ve target builded...")
    print("positive data length:", len(pos_data))
    print("Negative data ve target builded..")
    print("negative data length :", len(neg_data))
    data = pos_data + neg_data
    target_ = pos_target + neg_target
    print("Positive and Negative sets concatenated")
    print("data length :", len(data))

    le = LabelEncoder()
    le.fit(target_)
    target = le.transform(target_)
    print("Target labels transformed to number...")

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=24)
    print("Dataset splited into train and test parts...")
    print("train data length  :", len(X_train))
    print("train target length:", len(y_train))
    print()
    print("test data length  :", len(X_test))
    print("test target length:", len(y_test))
