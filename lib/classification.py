import pandas as pd 
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import svm

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
	adaboost for 1 and 5 star
''' 
df15 = df[((df.stars == 1)| (df.stars == 5))]
vectorizer = CountVectorizer(stop_words='english')
# matrix is a n* 2000+ matrx, with each words represented by integer
matrix = vectorizer.fit_transform(df15.text)

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


