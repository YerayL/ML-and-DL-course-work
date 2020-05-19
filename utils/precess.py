from nltk.corpus import stopwords
import string, re
from collections import Counter
import wordcloud
import seaborn as sns

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os

""" 
refer: https://www.kaggle.com/serkanpeldek/text-classification-with-embedding-conv1d
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


# text processing
class MakeString:
    def process(self, text):
        return str(text)


class ReplaceBy:
    def __init__(self, replace_by):
        # replace_by is a tuple contains pairs of replace and by characters.
        self.replace_by = replace_by

    def process(self, text):
        for replace, by in self.replace_by:
            text = text.replace(replace, by)
        return text


class LowerText:
    def process(self, text):
        return text.lower()


class ReduceTextLength:
    def __init__(self, limited_text_length):
        self.limited_text_length = limited_text_length
    def process(self, text):
        return text[:self.limited_text_length]


# word vector processing
class VectorizeText:
    def __init__(self):
        pass
    def process(self, text):
        return text.split()

class FilterPunctuation:

    def __init__(self):
        print("Punctuation Filter created...")
    def process(self, words_vector):
        reg_exp_filter_rule=re.compile("[%s]"%re.escape(string.punctuation))
        words_vector=[reg_exp_filter_rule.sub("", word) for word in words_vector]
        return words_vector

class FilterNonalpha:
    def __init__(self):
        print("Nonalpha Filter created...")
    def process(self, words_vector):
        words_vector=[word for word in words_vector if word.isalpha()]
        return words_vector

class FilterStopWord:
    def __init__(self, language):
        self.language=language
        print("Stopwords Filter created...")
    def process(self, words_vector):
        stop_words=set(stopwords.words(self.language))
        words_vector=[word for word in words_vector if not word in stop_words]
        return words_vector

class FilterShortWord:
    def __init__(self, min_length):
        self.min_length=min_length
        print("Short Words Filter created...")
    def process(self, words_vector):
        words_vector=[word for word in words_vector if len(word)>=self.min_length]
        return words_vector


# text processing
class TextProcessor:
    def __init__(self, processor_list):
        self.processor_list = processor_list
    def process(self, text):
        for processor in self.processor_list:
            text = processor.process(text)
        return text

class VocabularyHelper:
    def __init__(self, textProcessor):
        self.textProcessor=textProcessor
        self.vocabulary=Counter()
    def update(self, text):
        words_vector=self.textProcessor.process(text=text)
        self.vocabulary.update(words_vector)
    def get_vocabulary(self):
        return self.vocabulary


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

    print("正样本数量：", len(pos_data))
    print("负样本数量：", len(neg_data))
    data = pos_data + neg_data
    target_ = pos_target + neg_target
    print("总共数据：", len(data))

    # 标签转为数字，正1负0
    le = LabelEncoder()
    le.fit(target_)
    target = le.transform(target_)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=24)

    print("训练数据数：", len(X_train))
    print("训练标签数：", len(y_train))
    print("测试样本数：", len(X_test))
    print("测试标签数", len(y_test))

    # fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
    # axarr[0].set_title("Number of samples in train")
    # sns.countplot(x=y_train, ax=axarr[0])
    # axarr[1].set_title("Number of samples in test")
    # sns.countplot(x=y_test, ax=axarr[1])
    # plt.show()


    text_len = np.vectorize(len) # 向量化函数
    text_lengths = text_len(X_train) # [6975 4351 3120 ... 1869 4961 3808]


    mean_review_length = int(text_lengths.mean())
    print("平均评论长度:", mean_review_length)
    print("最小评论长度:", text_lengths.min())
    print("最大评论长度:", text_lengths.max())

    # 绘制文本长度的分布
    sns.distplot(a=text_lengths)

    # str()
    makeString = MakeString()

    # 过滤标点符号
    replace_by = [(".", " "), ("?", " "), (",", " "), ("!", " "), (":", " "), (";", " ")]
    replaceBy = ReplaceBy(replace_by=replace_by)

    # 小写
    lowerText = LowerText()

    # 限制文本长度
    FACTOR = 8
    reduceTextLength = ReduceTextLength(limited_text_length=mean_review_length * FACTOR)

    # 以空格分割
    vectorizeText = VectorizeText()

    # 过滤标点符号
    filterPunctuation = FilterPunctuation()
    # 过滤非字母字符
    filterNonalpha = FilterNonalpha()

    # 过滤停用词
    filterStopWord = FilterStopWord(language="english")

    # 过滤短词
    min_length = 2
    filterShortWord = FilterShortWord(min_length=min_length)

    # 预处理列表
    processor_list_1 = [makeString,
                        replaceBy,
                        lowerText,
                        reduceTextLength,
                        vectorizeText,
                        filterPunctuation,
                        filterNonalpha,
                        filterStopWord,
                        filterShortWord]

    # 文本处理器
    textProcessor1 = TextProcessor(processor_list=processor_list_1)

    # 随机看一个训练样本
    random_number = np.random.randint(0, len(X_train))
    print("Original Review:\n", X_train[random_number][:500])
    print("=" * 100)
    print("Processed Review:\n", textProcessor1.process(text=X_train[random_number][:500]))

    # 建立词汇表
    vocabularyHelper = VocabularyHelper(textProcessor=textProcessor1)
    for text in X_train:
        vocabularyHelper.update(text)
    vocabulary = vocabularyHelper.get_vocabulary()
    print("词汇表长度:", len(vocabulary))
    print("词汇表:", vocabulary)


    n = 10
    print("{} most frequented words in vocabulary:{}".format(n, vocabulary.most_common(n)))

    print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n - 1:-1]))

    vocabulary_list = " ".join([key for key, _ in vocabulary.most_common()])
    plt.figure(figsize=(15, 35))
    wordcloud_image = wordcloud.WordCloud(width=1000, height=1000,
                                          background_color='white',
                                          # stopwords = stopwords,
                                          min_font_size=10).generate(vocabulary_list)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(wordcloud_image)

    min_occurence = 2
    vocabulary = Counter({key: value for key, value in vocabulary.items() if value > min_occurence})

    print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n - 1:-1]))

    print("Length of vocabulary after removing words occurenced less than {} times:{}".format(min_occurence,
                                                                                              len(vocabulary)))


