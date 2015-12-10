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


f = open('diabetic_data_final1.csv')
f.readline() # skip the header
data = np.loadtxt(f, delimiter=",",dtype=np.str)
data = np.char.strip(data, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')

#print(data[2,14])
data = data[:,1:]
data_size = data.shape
print(data_size)

i = 0
for i in xrange(data_size[0]):

    if data[i,13] == '?':
        data[i,13] = '0'
    if data[i,14] == '?':
        data[i,14] = '0'
    if data[i,15] == '?':
        data[i,15] = '0'


data = data.astype(np.float)
np.random.shuffle(data)

tmp = data[0:10000,:]


data_size_tmp = tmp.shape
row = data_size_tmp[0]
attribute = data_size_tmp[1]

print("data size")
print(data_size_tmp)
print("\n")

train = tmp[:,:attribute-2]
tag = tmp[:,attribute-1]

# 5-fold cross validation
kf = cross_validation.KFold(len(tmp), n_folds=5)

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

#logregr_predict = logregr.predict(train)
#logregr_accuracy = logregr.score(train, tag)

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
