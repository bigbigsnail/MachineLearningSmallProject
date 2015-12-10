__author__ = 'bigbigsnail'

import os
from numpy.core.defchararray import strip
import scipy
import numpy as np

from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import f1_score

f = open('746Data_final(1).txt')
data = np.loadtxt(f, delimiter=",", dtype=int)

data_size = data.shape
row = data_size[0]
attribute = data_size[1]

i = 0
for i in xrange(row):
    if data[i,attribute-1] == -1:
        data[i,attribute-1] = 0

#print(data[:,20])
np.random.shuffle(data)

print("data size:")
print(data_size)
print("\n")

train = data[:,0:attribute-2]
tag = data[:,attribute-1]

kf = cross_validation.KFold(len(data), n_folds=5)

# Naive Bayes
print("Gaussian Naive Bayes")
gnb = GaussianNB()
gnb.fit(train,tag)
gnb_predict = cross_validation.cross_val_predict(gnb, train, tag, cv=kf)
gnb_accuracy = cross_validation.cross_val_score(gnb, train, tag, cv=kf)

gnb_micro_fmeasure = f1_score(tag, gnb_predict, average='micro')
gnb_macro_fmeasure = f1_score(tag, gnb_predict, average='macro')

print(gnb_accuracy)
print("The micro f-measure of Gaussian Naive Bayes is:")
print(gnb_micro_fmeasure)
print("The macro f-measure of Gaussian Naive Bayes is:")
print(gnb_macro_fmeasure)
print('\n')

#print(tag)
#print(gnb_predict)


# Logistic Regression
print("Logistic Regression")
logregr = linear_model.LogisticRegression()
logregr.fit(train, tag)

logregr_predict = cross_validation.cross_val_predict(logregr, train, tag, cv=kf)
logregr_accuracy = cross_validation.cross_val_score(logregr, train, tag, cv=kf)

logregr_micro_fmeasure = f1_score(tag, logregr_predict, average='micro')
logregr_macro_fmeasure = f1_score(tag, logregr_predict, average='macro')

print(logregr_accuracy)
print("The micro f-measure of Logistic Regression is:")
print(logregr_micro_fmeasure)
print("The macro f-measure of Logistic Regression is:")
print(logregr_macro_fmeasure)
print('\n')


# SVM
print('SVM')
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train, tag)
svc_predict = cross_validation.cross_val_predict(clf, train, tag, cv=kf)
svc_accuracy = cross_validation.cross_val_score(clf, train, tag, cv=kf)

svc_micro_fmeasure = f1_score(tag, svc_predict, average='micro')
svc_macro_fmeasure = f1_score(tag, svc_predict, average='macro')

print(svc_accuracy)
print("The micro f-measure of SVM is:")
print(svc_micro_fmeasure)
print("The macro f-measure of SVM is:")
print(svc_macro_fmeasure)
