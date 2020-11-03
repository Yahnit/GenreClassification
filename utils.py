import numpy as npy
import matplotlib.pyplot as plt
from stemming.porter2 import stem
from nltk.corpus import wordnet
import csv
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import findspark
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.functions import length
from pyspark.mllib.feature import HashingTF, IDF
from sklearn import cross_validation, svm
from stemming.porter2 import stem
from sklearn.metrics import confusion_matrix
import sys
from sklearn.pipeline import Pipeline
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing
import sklearn

genre_types = ["Rock", "Jazz", "Pop", "Country", "Hip-Hop"]
unwanted_genres = ["Folk", "R&B", "Indie", "Electronic", "Metal"]
strip_items = ['[]', '()']
invalids = ["Alkebulan", "zora sourit", "Other", "Not Available", ""]
stopwords = set(stopwords.words('english'))
replace_items = ['[^\w\s]', 'chorus', ':', ',', 'verse', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
inp_file = 'data/lyrics.csv'
out_file = 'data/lyrics_out.csv'
dict_genres={'Rock':1, 'Country':2, 'Hip-Hop':3, 'Pop':4, 'Jazz':5}
countVec = TfidfVectorizer(stop_words = 'english', sublinear_tf=True)
format, data_analy_inp = "com.databricks.spark.csv", r'data/lyrics_final.csv'
final_output = 'data/lyrics_final.csv'
text, cnt, max_cnt = 'lyrics', 'word_count', 1000

def isValid(line):
    for i in invalids:
        if line[4] == i:
            return False
    return True

def deleteStopWords(words):
    for word in words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words('english'):
            words.append(word)
    return words

def initialProcess():
    data = pd.read_csv('data/lyrics_out.csv')
    data[cnt] = data[text].str.split( ).str.len()
    data[text] = data[text].str.lower()
    return data

def genre(x, type):
    if type not in x:
        return ("Non " + str(type))
    else:
        return type

def getData(file):
    data = pd.read_csv(file)
    data.replace('?', -9999999, inplace=True)
    data.drop(['index'],1, inplace=True)
    return data

def plotAvgWordsGraph(y_pos, average_word_list, all_genres):
    plt.bar(y_pos, average_word_list, align='center', alpha=0.5)
    plt.xticks(y_pos, all_genres)
    plt.ylabel('Average number of words used')
    plt.title('Genres')
    plt.show()

def plotUniqueWordsGraph(y_pos, unique_word_list, all_genres):
    plt.bar(y_pos, unique_word_list, align='center', alpha=0.5)
    plt.xticks(y_pos, all_genres)
    plt.ylabel('Number of unique words used')
    plt.title('Genres')
    plt.show()

def getTrainTestData(lyrics, labels):
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(lyrics,labels,test_size=0.2)
    x_train, x_test = preprocessing.scale(preprocessing.normalize(countVec.fit_transform(x_train).toarray())), preprocessing.scale(preprocessing.normalize(countVec.transform(x_test).toarray()))
    return x_train, x_test, y_train, y_test

def init():
    findspark.init("/opt/spark")
    cntxt = SparkContext(conf=SparkConf())
    sql = SQLContext(cntxt)
    data = cntxt.textFile(data_analy_inp)
    data = sql.read.format(format).option("header", "true").load(data_analy_inp)
    list_of_genres=data.select('Genre').distinct().rdd.map(lambda r: r[0]).collect()
    genres_tot=[i.Genre for i in data.select('Genre').distinct().collect()]
    return genres_tot, data

def finalProcess(data):
    data = data[data[cnt] > max_cnt/10]
    data = data.groupby('genre').head(max_cnt)
    data = data.replace({'\n': ' '}, regex=True)
    data[text] = data[text].str.lower().replace('[^\w\s]','')
    del data['song'],data['year'],data['artist'],data[cnt]
    return data

def binClassifierProcess(type):
    data = pd.read_csv(out_file)
    data[cnt] = data[text].str.split().str.len()
    data["genre values"] = data["genre"].apply(lambda x: genre(x, type))
    data = data[data[cnt] > max_cnt/10]
    data = data.replace({'\n': ' '}, regex=True)
    data["lyrics"] = data[text].str.lower().replace('[^\w\s]','')
    del data['song'],data['year'],data['artist'],data[cnt],data['genre']
    return data
