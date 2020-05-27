import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import os

#这里填数据集名字就行了
dataset = "movie"

data_path = "data"
train = pd.read_csv(os.path.join(data_path, dataset, "train.tsv"), sep='\t',header=None,names=["label", "data"])
test = pd.read_csv(os.path.join(data_path, dataset, "test.tsv"), sep='\t',header=None,names=["label", "data"])
x_train = train['data']
y_train = train['label']
x_test = test['data']
y_test = test['label']

#提取文本计数特征
#对文本的单词进行计数，包括文本的预处理, 分词以及过滤停用词
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
x_test_counts = count_vect.transform(x_test)
print(x_train_counts.shape)
print(x_train_counts)

print(count_vect.vocabulary_.get(u'good'))

#提取TF-IDF特征-word
#将各文档中每个单词的出现次数除以该文档中所有单词的总数：这些新的特征称之为词频tf。
tfidf_transformer = TfidfVectorizer(analyzer='word',max_features=50000)
tfidf_transformer.fit(x_train)
x_train_tfidf_word = tfidf_transformer.transform(x_train)
x_test_tfidf_word = tfidf_transformer.transform(x_test)
print(x_train_tfidf_word.shape)

#提取TF-IDF特征-ngram
#将各文档中每个单词的出现次数除以该文档中所有单词的总数：这些新的特征称之为词频tf。
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfVectorizer(analyzer='word', ngram_range=(2,3),max_features=50000)
tfidf_transformer.fit(x_train)
x_train_tfidf_ngram = tfidf_transformer.transform(x_train)
x_test_tfidf_ngram = tfidf_transformer.transform(x_test)
print(x_train_tfidf_ngram.shape)

#合并特征（特征组合与特征选择）
train_features=x_train_counts
test_features=x_test_counts
train_features = hstack([x_train_counts,x_train_tfidf_word, x_train_tfidf_ngram])
test_features = hstack([x_test_counts,x_test_tfidf_word ,x_test_tfidf_ngram])
train_features.shape


# 机器学习算法合集
# 1.朴素贝叶斯
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()

# 2.KNN
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

# 3.决策树
# from sklearn.tree import DecisionTreeRegressor
# clf = DecisionTreeRegressor()

# 4.AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()

# 5.Bagging
# from sklearn.ensemble import BaggingClassifier
# clf = BaggingClassifier()

# 6.GradientBoosting
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier()

# 7.RandomForest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()

# 8.ExtraTrees
# from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier()

# 9.SVM or LR
from sklearn.linear_model import SGDClassifier
# SGDClassifier是一系列采用了梯度下降来求解参数的算法的集合，默认是SVM 也可以lr, 支持多分类
clf = SGDClassifier(alpha=0.001,
                    loss='hinge',    #hinge代表SVM，log是逻辑回归
                    early_stopping=True,
                    eta0=0.001,
                    learning_rate='adaptive', #constant、optimal、invscaling、adaptive
                    max_iter=100
                   )


#打乱数据，训练
from sklearn.utils import shuffle
train_features,y_train=shuffle(train_features,y_train )

clf.fit(train_features, y_train)

#测试过程
predict = clf.predict(test_features)

#测试集的评估
print(np.mean(predict == y_test))

#KNN 0.53
#lr 0.8575
#svm 0.8675
#NB 0.8275
#决策树 0.6575
#Adaboost 0.7675
#Bagging 0.72
#GradientBoosting 0.795
#RandomForest 0.835
#ExtraTrees 0.855
