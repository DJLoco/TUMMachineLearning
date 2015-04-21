# for converting byte to unsigned integer
import struct

import numpy

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

import pylab


# read training file ########################################
nrTrain = 800
y = numpy.zeros(nrTrain)
X = numpy.zeros((nrTrain, 784))

# read labels
with open("train-labels.idx1-ubyte", "rb") as fileLabels:
    byte = fileLabels.read(8)

    for n in range(1, nrTrain):
        byte = fileLabels.read(1)
        value = struct.unpack('B', byte)
        y[n] = value[0]


# read training set image file
with open("train-images.idx3-ubyte", "rb") as fileImages:
    byte = fileImages.read(16)

    for n in range(1, nrTrain):
        for i in range(1, 784):
            byte = fileImages.read(1)
            value = struct.unpack('B', byte)
            X[n, i] = value[0]


# read testing file ############################################
nrTest = 20
TestY = numpy.zeros(nrTest)
TestX = numpy.zeros((nrTest, 784))

# read labels
with open("t10k-labels.idx1-ubyte", "rb") as fileLabels:
    byte = fileLabels.read(8)

    for n in range(1, nrTest):
        byte = fileLabels.read(1)
        value = struct.unpack('B', byte)
        TestY[n] = value[0]


# read training set image file
with open("t10k-images.idx3-ubyte", "rb") as fileImages:
    byte = fileImages.read(16)

    for n in range(1, nrTest):
        for i in range(1, 784):
            byte = fileImages.read(1)
            value = struct.unpack('B', byte)
            TestX[n, i] = value[0]


# start training #####################################
clf = OneVsOneClassifier(LinearSVC()).fit(X, y)

# google method
from sklearn.svm import SVC
# scikit-learn dimension reduction
from sklearn.decomposition import PCA

# scikit-learn dataset processing utils
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
pca = PCA(n_components=80)


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1e-2, 1e-3],
                     'C': [10, 100, 1000]}, {'kernel': ['poly'],
                                             'degree': [5, 9], 'C': [1, 10]}]

from sklearn.grid_search import GridSearchCV
svm = GridSearchCV(SVC(), tuned_parameters, cv=3, verbose=2).fit(X, y)
# testing ################################################
print(clf.predict(TestX))
print(svm.predict(TestX))
print(TestY[0:20])

pylab.imshow(numpy.reshape(TestX[5], (28, 28)))
pylab.show()
