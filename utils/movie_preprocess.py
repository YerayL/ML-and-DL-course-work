from nltk.corpus import stopwords
import string, re
from collections import Counter
import wordcloud
import seaborn as sns
import regex as re
import numpy as np  # linear algebra


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os

""" 
refer: https://www.kaggle.com/serkanpeldek/text-classification-with-embedding-conv1d
采取的预处理：
    小写
    过滤标点符号
    过滤非字母字符
    限制文本长度
    过滤停用词
    过滤短词(<=2)
    过滤低频词(<=2)
"""

review_dataset_path = "../raw data/review_polarity/txt_sentoken"
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
    #  Counter({'film': 7098, 'movie': 4432, 'one': 4429, 'like': 2842,...})

    # 看一下频率最高的前10个词和最低的10个词
    n = 10
    print("{} most frequented words in vocabulary:{}".format(n, vocabulary.most_common(n)))

    print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n - 1:-1]))

    # 词云
    vocabulary_list = " ".join([key for key, _ in vocabulary.most_common()])
    plt.figure(figsize=(15, 35))
    wordcloud_image = wordcloud.WordCloud(width=1000, height=1000,
                                          background_color='white',
                                          # stopwords = stopwords,
                                          min_font_size=10).generate(vocabulary_list)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(wordcloud_image)

    #过滤低频词
    min_occurence = 2
    vocabulary = Counter({key: value for key, value in vocabulary.items() if value > min_occurence})

    print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n - 1:-1]))

    print("Length of vocabulary after removing words occurenced less than {} times:{}".format(min_occurence,
                                                                                              len(vocabulary)))

    # 过滤不在词典的
    class FilterNotInVocabulary:
        def __init__(self, vocabulary):
            self.vocabulary = vocabulary

        def process(self, words_vector):
            words_vector = [word for word in words_vector if word in self.vocabulary]
            return words_vector


    # 连起来
    class JoinWithSpace:
        def __init__(self):
            pass

        def process(self, words_vector):
            return " ".join(words_vector)


    filterNotInVocabulary = FilterNotInVocabulary(vocabulary=vocabulary)
    joinWithSpace = JoinWithSpace()
    processor_list_2 = [makeString,
                        replaceBy,
                        lowerText,
                        reduceTextLength,
                        vectorizeText,
                        filterPunctuation,
                        filterNonalpha,
                        filterStopWord,
                        filterShortWord,
                        filterNotInVocabulary,
                        joinWithSpace
                        ]
    textProcessor2 = TextProcessor(processor_list=processor_list_2)

    review = X_train[np.random.randint(0, len(X_train))]
    print("Original Text:\n", review[:500])
    processed_review = textProcessor2.process(review[:500])
    print("=" * 100)
    print("Processed Text:\n", processed_review)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=24)
    print("Dataset splited into train and test parts...")
    print("train data length  :", len(X_train))
    print("train target length:", len(y_train))
    print()
    print("test data length  :", len(X_test))
    print("test target length:", len(y_test))

    """去除符号"""
    re_punct = r"[\[\]\{\}\:\;\'\"\,\<\.\>\/\?\\\|\`\!\@\#\$\%\^\&\*\(\)\-\_\=\+]"
    for i,j in enumerate(X_train):
        X_train[i] = re.sub(re_punct, "", j)
        X_train[i] = X_train[i].replace('\n', '')
    for i,j in enumerate(X_test):
        X_test[i] = re.sub(re_punct, "", j)
        X_test[i] = X_train[i].replace('\n', '')


    import random

    X = list(zip(X_train,y_train))

    Y = list(zip(X_test,y_test))
    print(Y)
    random.shuffle(X)
    print(Y)
    random.shuffle(Y)

    X_train = [i for i,j in X]
    y_train = [j for i,j in X]
    X_test = [i for i,j in Y]
    y_test = [j for i,j in Y]


    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'a_name': y_train[:1400], 'b_name': X_train[:1400]})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("train.tsv", index=False, sep='\t', header=False)

    dataframe = pd.DataFrame({'a_name': y_train[1400:], 'b_name': X_train[1400:]})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("dev.tsv", index=False, sep='\t', header=False)

    dataframe = pd.DataFrame({'a_name': y_test, 'b_name': X_test})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("test.tsv", index=False, sep='\t', header=False)