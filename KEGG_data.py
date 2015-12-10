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


f = open('RelationNetworkDirected.data')
# f.readline() # skip the header
data = np.loadtxt(f, delimiter=",", dtype=np.str)

data[:, 0] = np.char.strip(data[:, 0], '0123456789')

# Change class name
class_list = []
class_list.append(1)
data_size = data.shape
i = 1
count = 0

for i in xrange(data_size[0]):
    if data[i,0] != data[i-1,0]:
        count += 1
        class_list.append(count)
    else:
        class_list.append(count)

#print(class_list[0:10])

i = 0
row = data_size[1]
for i in xrange(data_size[0]):
    data[i,0] = class_list[i]

#print(data[:,0])

data = data.astype(np.float)

tmp = data[0:10000,:]
np.random.shuffle(tmp)

train = tmp[:,1:]
tag = tmp[:,0]
data_size_tmp = tmp.shape

print("Data size:")
print(data_size_tmp)
print("\n")

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


'''
# Linear Regression
regr = linear_model.LinearRegression()
regr.fit(train,tag)
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(train) - tag) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.10f' % regr.score(train, tag))


regr_predict = cross_validation.cross_val_predict(regr, train, tag, cv=kf)
regr_accuracy = cross_validation.cross_val_score(regr, train, tag, cv=kf)

regr_micro_fmeasure = f1_score(tag, regr_predict, average='micro')
regr_macro_fmeasure = f1_score(tag, regr_predict, average='macro')

print(regr_accuracy)
print("The micro f-measure of Linear Regression is:")
print(regr_micro_fmeasure)
print("The macro f-measure of Linear Regression is:")
print(regr_macro_fmeasure)
print('\n')
'''
