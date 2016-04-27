import pandas as pd 
import numpy as np
import heapq
import matplotlib.pyplot as plt


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from wordcloud import WordCloud, STOPWORDS

df = pd.read_pickle('./data/review.pkl')
vectorizer = CountVectorizer(stop_words='english')
# matrix is a n* 2000+ matrx, with each words represented by integer
matrix = vectorizer.fit_transform(df.text)

X = matrix
y = df.stars.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

'''
	adaboost
'''
 
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
bdt.fit(X_train, y_train)

z = bdt.predict(X_test)

count = 0
for i in range(len(z)):
	if z[i] not in range(y_test[i]-1,y_test[i]+2):
		count += 1
print 'adaboost'
print 'error: {}'.format(count)
print 'total test: {}'.format(len(y_test))
print 



'''
	svm
''' 

clf = svm.SVC()
clf.fit(X_train, y_train)
z = clf.predict(X_test)
count = 0
for i in range(len(z)):
	if z[i] != y_test[i]:
		count += 1
print 'SVM15'
print 'error: {}'.format(count)
print 'total test: {}'.format(len(y_test))
print 



'''
	adaboost for 1 and 5 star
''' 
df15 = df[((df.stars == 1)| (df.stars == 5))]
dfGood = df[(df.stars == 5)]
dfBad = df[(df.stars == 1)]

vectorizer = CountVectorizer(stop_words='english')
# matrix is a n* 2000+ matrx, with each words represented by integer
matrix = vectorizer.fit_transform(df15.text)
#print vectorizer.get_feature_names()
#print zip(vectorizer.get_feature_names(), np.asarray(matrix.sum(axis=0)).ravel())
X = matrix
y = df15.stars.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
 
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
bdt.fit(X_train, y_train)

z = bdt.predict(X_test)

count = 0
for i in range(len(z)):
	if z[i] != y_test[i]:
		count += 1
print 'adaboost15'
print 'error: {}'.format(count)
print 'total test: {}'.format(len(y_test))
print 

vocab = vectorizer.vocabulary_ # vocab of all appeared words in corpus
importances = bdt.feature_importances_
index = [i for i, e in enumerate(importances) if e != 0]

dictionary = {}
wordImptDict = {}
imptWordDict = {}
for value in index:
	for word, ind in vocab.iteritems():
		if value == ind:
			dictionary[word]=value
			wordImptDict[word] = importances[value]


nMostImpt = 10

words = heapq.nlargest(nMostImpt, wordImptDict, key = wordImptDict.get)
words = [word.encode() for word in words] # array of important words
impt = [wordImptDict[word] for word in words  ] # array of importances 


###################################################################################
# plot bar plot of importances against words
plt.figure()
plt.title("Word Importances")
plt.xticks(range(len(impt)), words)
plt.bar(range(len(impt)), impt, align='center')
#plt.show()
###################################################################################
# Important Words of Good Reviews
goodDf = df[(df.stars == 5)]
goodReviews = goodDf.text
goodText ='' 
for review in goodReviews:
	for word in review.split():
		if word in wordImptDict:
			goodText = goodText+' '+ word
wordcloud = WordCloud(stopwords=STOPWORDS).generate(goodText)
plt.figure()
plt.title("Important Words of Good Reviews")
plt.imshow(wordcloud)
plt.axis("off")
#plt.show()
###################################################################################
# Important Words of Bad Reviews
badDf = df[(df.stars == 1)]
badReviews = badDf.text
badText ='' 
for review in badReviews:
	for word in review.split():
		if word in wordImptDict:
			badText = badText+' '+ word
wordcloud = WordCloud(stopwords=STOPWORDS).generate(badText)
plt.figure()
plt.title("Important Words of Good Reviews")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()