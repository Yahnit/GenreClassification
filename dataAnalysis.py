from operator import add
import string
from utils import *

list_avg_wrds, unq_wrds_lst, stop_word_list =[], [], []
f = open("results/dataanalysis_output.txt", "w")
genres_tot, data = init()

for type in genres_tot:
    genre_data = data.filter(data.genre == type)
    list_avg_wrds.append(genre_data.select(text).rdd.flatMap(lambda p: p[0].split(" ")).count()/data.filter(data.genre == type).count())
    unq_wrds_lst.append(genre_data.select(text).rdd.flatMap(lambda p: p[0].split(" ")).map(lambda p: (p, 1)).reduceByKey(add).count()/data.filter(data.genre == type).count())

f.close()

y_pos = np.arange(len(genres_tot))
plotAvgWordsGraph(y_pos, list_avg_wrds, genres_tot)
plotUniqueWordsGraph(y_pos, unq_wrds_lst, genres_tot)
