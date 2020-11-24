from utils import *
import numpy

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

def BinaryClassifier(genre):
	labels, lyrics = [], []
	filename = "data/lyrics_%s.csv"%(genre)
	data = getData(filename)
	arr_data = data.iterrows()
	belongs, not_belongs = genre, "Non " + str(genre)
	dict_genres={belongs:1, not_belongs:2}
	count = 0
	for i, j in arr_data:
		count += 1
		if not i%100:
			labels.append(dict_genres[j[1]])
			lyrics.append(j[0])
	x_train, x_test, y_train, y_test = getTrainTestData(lyrics, labels)
	print(count)
	print(len(labels))
	print("x_train: %s, x_test: %s, y_train: %s, y_test: %s"%(len(x_train),len(x_test),len(y_train),len(y_test)))

	if classifier_type == "svm":
		classifier = SVM()

	if classifier_type == "mnb":
		classifier = MultinomialNBC()

	if classifier_type == "bnb":
		classifier = BernoulliNBC()

	if classifier_type == "lr":
		classifier = LogisticRegressionC()

	if classifier_type == "dt":
		classifier = DecisionTreeC()

	if classifier_type == "mlp":
		classifier = MultiLayerPerceptronC()

	if classifier_type == "rf":
		classifier = RandomForestC()

	classifier.fit(x_train,y_train)
	accuracy = classifier.score(x_test,y_test)
	output = str(classifier_type) + " accuracy : " + str(accuracy)
	f = open("results/binaryclassifier.txt", "w")
	print("For genre : " + str(genre))
	f.write("For genre : " + str(genre))
	print(output)
	f.write(output)
	f.close()

classifier_type = sys.argv[1]
for genre in genre_types:
	BinaryClassifier(genre)
