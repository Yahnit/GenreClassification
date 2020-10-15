import numpy as np
import time
from utils import *

def SVM():
	return Pipeline([('vect', TfidfVectorizer(analyzer=lambda x: x)),('clf', sklearn.svm.SVC(kernel = 'linear'),)])

def MultinomialNBC():
	return MultinomialNB()

def BernoulliNBC():
	return BernoulliNB()

def LogisticRegressionC():
	return LogisticRegression(solver="liblinear", multi_class="ovr")

def DecisionTreeC():
	return DecisionTreeClassifier()

def MultiLayerPerceptronC():
	return MLPClassifier()

def RandomForestC():
	return RandomForestClassifier(n_estimators=100,max_features="sqrt")

labels, lyrics = [], []
data = getData(final_output)
arr_data = data.iterrows()
f = open("results/multiclass_output.txt", "w")
for i, j in arr_data:
	lyrics.append(j[1])
	labels.append(j[0])

lyrics = [" ".join(ln) for ln in [[stem(wd) for wd in ln.split(" ")] for ln in lyrics]]
x_train, x_test, y_train, y_test = getTrainTestData(lyrics, labels)
print("x_train: %s, x_test: %s, y_train: %s, y_test: %s"%(len(x_train),len(x_test),len(y_train),len(y_test)))

if sys.argv[1] == "svm":
	classifier = SVM()

if sys.argv[1] == "mnb":
	classifier = MultinomialNBC()

if sys.argv[1] == "bnb":
	classifier = BernoulliNBC()

if sys.argv[1] == "lr":
	classifier = LogisticRegressionC()

if sys.argv[1] == "dt":
	classifier = DecisionTreeC()

if sys.argv[1] == "mlp":
	classifier = MultiLayerPerceptronC()

if sys.argv[1] == "rf":
	classifier = RandomForestC()

classifier.fit(x_train,y_train)
accuracy = classifier.score(x_test,y_test)
output = str(sys.argv[1]) + " accuracy : " + str(accuracy)
print(output)
f.write(output)
f.close()
