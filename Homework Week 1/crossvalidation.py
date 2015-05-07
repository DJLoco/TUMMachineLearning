__author__ = 'nanliu'
__project__ = 'crossvalidation'

# for converting byte to unsigned integer
import struct
import numpy
import pylab
from sklearn.externals.six import StringIO
#import pydot
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# read training file #####################################################
nrTrain = 10000
y_Train = numpy.zeros(nrTrain)
X_Train = numpy.zeros((nrTrain, 784))

# read training label files ###############################################
with open("train-labels.idx1-ubyte", "rb") as fileLabels:
    byte = fileLabels.read(8)

    for n in range(0, nrTrain):
        byte = fileLabels.read(1)
        value = struct.unpack('B', byte)
        y_Train[n] = value[0]


# read training image files ################################################
with open("train-images.idx3-ubyte", "rb") as fileImages:
    byte = fileImages.read(16)

    for n in range(0, nrTrain):
        image = []
        for i in range(1, 784):
            byte = fileImages.read(1)
            value = struct.unpack('B', byte)
            X_Train[n, i] = value[0]

# preprocess the dataset ##################################################
min_max_scaler = preprocessing.MinMaxScaler()
X_Train = min_max_scaler.fit_transform(X_Train)

# use cross validation for decision tree ##################################
kf = cross_validation.KFold(nrTrain, n_folds= 3)
score_dt = 0
score_rf = 0
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_Train[train_index], X_Train[test_index]
    y_train, y_test = y_Train[train_index], y_Train[test_index]
    # decision tree with cross validation
    clf_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None,max_features=None)
    clf_dt  = clf_dt.fit(X_train, y_train)
    score_dt += accuracy_score(y_test, clf_dt.predict(X_test))

    # random forests with cross validation
    clf_rf = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth= None,max_features='auto')
    clf_rf = clf_rf.fit(X_train, y_train)
    score_rf += accuracy_score(y_test, clf_rf.predict(X_test))

print('decision tree with cross validation:')
print(score_dt)

print('random forests with cross validation:')
print(score_rf)
