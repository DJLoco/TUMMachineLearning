import struct
import numpy

# # read training file ########################################
nrTrain = 1000
y = numpy.zeros(nrTrain)
X = numpy.zeros((nrTrain, 784))

# read labels
with open("train-labels.idx1-ubyte", "rb") as fileLabels:
    byte = fileLabels.read(8)

    for n in range(0, nrTrain):
        byte = fileLabels.read(1)
        value = struct.unpack('B', byte)
        y[n] = value[0]

# read training set image file
with open("train-images.idx3-ubyte", "rb") as fileImages:
    byte = fileImages.read(16)

    for n in range(0, nrTrain):
        for i in range(0, 784):
            byte = fileImages.read(1)
            value = struct.unpack('B', byte)
            X[n, i] = value[0]


# # read testing file ############################################
nrTest = 2000
TestY = numpy.zeros(nrTest)
TestX = numpy.zeros((nrTest, 784))

# read labels
with open("t10k-labels.idx1-ubyte", "rb") as fileLabels:
    byte = fileLabels.read(8)

    for n in range(0, nrTest):
        byte = fileLabels.read(1)
        value = struct.unpack('B', byte)
        TestY[n] = value[0]


# read training set image file
with open("t10k-images.idx3-ubyte", "rb") as fileImages:
    byte = fileImages.read(16)

    for n in range(0, nrTest):
        for i in range(0, 784):
            byte = fileImages.read(1)
            value = struct.unpack('B', byte)
            TestX[n, i] = value[0]

# # visualization #########################################
# import pylab
# i = 1900
# print(TestY[i])
# pylab.imshow(numpy.reshape(TestX[i], (28, 28)))
# pylab.show()


# # preprocessing #####################################

# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()

# X = min_max_scaler.fit_transform(X)
# TestX = min_max_scaler.transform(TestX)

# X = X/255.0*2 - 1
# TestX = TestX/255.0*2 - 1

# # start training #####################################
from sklearn import svm

rbfClf = svm.SVC().fit(X, y)    # default svm
# rbfClf = svm.SVC(C=2.8, gamma= 0.0073).fit(X, y)    # default svm
linClf = svm.SVC(kernel='linear').fit(X, y)
polyClf = svm.SVC(kernel='poly').fit(X, y)

# # testing ################################################
from sklearn.metrics import accuracy_score
print(accuracy_score(TestY, rbfClf.predict(TestX)))
print(accuracy_score(TestY, linClf.predict(TestX)))
print(accuracy_score(TestY, polyClf.predict(TestX)))
