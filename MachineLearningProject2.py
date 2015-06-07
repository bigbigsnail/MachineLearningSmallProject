import os
import scipy
import numpy as np

from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import f1_score


f = open("lung-cancer.data")
# f.readline() # skip the header
data = np.loadtxt(f, delimiter=",", dtype=int)

# random function for random data
np.random.shuffle(data)

train = data[:, 1:]
tag = data[:, 0]

# 5-fold cross validation
kf = cross_validation.KFold(len(data), n_folds=5)

# SVM
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train, tag)
svc_predict = cross_validation.cross_val_predict(clf, train, tag, cv=kf)
svc_accuracy = cross_validation.cross_val_score(clf, train, tag, cv=kf)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(train, tag)
gnb_predict = cross_validation.cross_val_predict(gnb, train, tag, cv=kf)
gnb_accuracy = cross_validation.cross_val_score(gnb, train, tag, cv=kf)

# Weighted empirical error
svc_correct = []
gnb_correct = []

# compare results
for i in range(len(data)):
    if svc_predict[i] == tag[i]:
        svc_correct.append(1)
    elif svc_predict[i] != tag[i]:
        svc_correct.append(0)

    if gnb_predict[i] == tag[i]:
        gnb_correct.append(1)
    elif gnb_predict[i] != tag[i]:
        gnb_correct.append(0)

#print(svc_correct)

# Weighted empirical error
svc_weight = []
gnb_weight = []
length = len(data)
j = 1

# compute weight
while j <= 3:
    w1 = 0.0
    w2 = 0.0

    for i in range(len(data)):
        if tag[i] == j:
            if svc_correct[i] == 1:
                w1 += 1.0
            if gnb_correct[i] == 1:
                w2 += 1.0

    w1 = length / w1
    w2 = length / w2
    svc_weight.append(w1)
    gnb_weight.append(w2)

    j += 1

#print(svc_weight)
#print(gnb_weight)

# compute error
index = 1
tmp1 = 0.0
tmp2 = 0.0
svc_error = 0.0
gnb_error = 0.0

while (index <= 3):

    for j in range(len(data)):
        if tag[i] == index:
            if svc_correct[i] == 0:
                tmp1 = svc_weight[index-1]
                svc_error += tmp1
            if gnb_correct[i] == 0:
                tmp2 = gnb_weight[index-1]
                gnb_error += tmp2

    index += 1

svc_error = svc_error / length
gnb_error = gnb_error / length


print("Weighted empirical error of SVM is: ")
print(svc_error)
print("Weighted empirical error of Gaussian Naive Bayes is: ")
print(gnb_error)

# F-measure
svc_micro_fmeasure = f1_score(tag, svc_predict, average='micro')
svc_macro_fmeasure = f1_score(tag, svc_predict, average='macro')
gnb_micro_fmeasure = f1_score(tag, gnb_predict, average='micro')
gnb_macro_fmeasure = f1_score(tag, gnb_predict, average='macro')


print("The micro f-measure of SVM is:")
print(svc_micro_fmeasure)
print("The macro f-measure of SVM is:")
print(svc_macro_fmeasure)
print("The micro f-measure of Gaussian Naive Bayes is:")
print(gnb_micro_fmeasure)
print("The macro f-measure of Gaussian Naive Bayes is:")
print(gnb_macro_fmeasure)
print('\n')

