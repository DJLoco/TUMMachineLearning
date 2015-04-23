__author__ = 'nanliu'
__project__ = 'randomforests'

# for converting byte to unsigned integer
import struct
import numpy
import pylab
from sklearn.externals.six import StringIO
#import pydot
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# read training file #####################################################
nrTrain = 10000
y_Train = numpy.zeros(nrTrain)
X_Train = numpy.zeros((nrTrain, 784))

nrTest = 10000
y_Test = numpy.zeros(nrTest)
X_Test = numpy.zeros((nrTest, 784))

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

# read test label files ###################################################
with open("t10k-labels.idx1-ubyte", "rb") as fileLabels:
    byte = fileLabels.read(8)

    for n in range(0, nrTest):
        byte = fileLabels.read(1)
        value = struct.unpack('B', byte)
        y_Test[n] = value[0]

# read test image files ####################################################
with open("t10k-images.idx3-ubyte", "rb") as fileImages:
    byte = fileImages.read(16)

    for n in range(0, nrTest):
        image = []
        for i in range(1, 784):
            byte = fileImages.read(1)
            value = struct.unpack('B', byte)
            X_Test[n, i] = value[0]

# preprocess the dataset ####################################################
min_max_scaler = preprocessing.MinMaxScaler()
X_Train = min_max_scaler.fit_transform(X_Train)
X_Test = min_max_scaler.fit_transform(X_Test)

# construct random forests classifier wrt. n_estimators#######################################
score_8 = 0
score_10 = 0
score_12 = 0

clf = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_10 = accuracy_score( y_Test, clf.predict(X_Test))


clf = RandomForestClassifier(n_estimators=8,criterion='gini',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_8 = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=12,criterion='gini',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_8 = accuracy_score( y_Test, clf.predict(X_Test))

if score_8 >= score_10 and score_8 >= score_12:
    print('the best n_estimators is 8')
    print(score_8)
if score_10 >= score_12 and score_10 >= score_8:
    print('the best n_estimators is 10')
    print(score_10)
if score_12 >= score_8 and score_12 >= score_10:
    print('the best n_estimators is 12')
    print(score_12)

# construct random forests classifier wrt. criterion ###########################################
score_gini = 0
score_entropy = 0

clf = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_gini = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_entropy = accuracy_score( y_Test, clf.predict(X_Test))

if score_gini >= score_entropy:
    print('the best criterion is gini')
    print(score_gini)
else:
    print('the best criterion is entropy')
    print(score_entropy)

# train a decision tree wrt. max_depth ######################################################
score_none = 0
score_int = 0

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_none = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= 10,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_int = accuracy_score( y_Test, clf.predict(X_Test))

if score_none >= score_int:
    print('the best max_depth is None')
    print(score_none)
else:
    print('the best max_depth is 10')
    print(score_int)

# train a decision tree wrt. max_features ################################################
score_auto = 0
score_none = 0
score_int = 0
score_float = 0
score_log = 0

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features='auto')
clf = clf.fit(X_Train, y_Train)
score_auto = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features=300)
clf = clf.fit(X_Train, y_Train)
score_int = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features=0.5)
clf = clf.fit(X_Train, y_Train)
score_float = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features=None)
clf = clf.fit(X_Train, y_Train)
score_none = accuracy_score( y_Test, clf.predict(X_Test))

clf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth= None,max_features='log2')
clf = clf.fit(X_Train, y_Train)
score_log = accuracy_score( y_Test, clf.predict(X_Test))

if score_none >= max(score_int, score_float, score_auto, score_log ):
    print('the best max_features is None')
    print(score_none)
if score_int >= max(score_none, score_float, score_auto, score_log ):
    print('the best max_features is int')
    print(score_int)
if score_float >= max(score_int, score_none, score_auto, score_log ):
    print('the best max_features is float')
    print(score_float)
if score_auto >= max(score_int, score_float, score_none, score_log ):
    print('the best max_features is sqrt')
    print(score_auto)
if score_log >= max(score_int, score_float, score_auto, score_none ):
    print('the best max_features is log')
    print(score_log)

clf = clf.fit(X_Train, y_Train)
score_none = accuracy_score( y_Test, clf.predict(X_Test))

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10,max_features=None)
clf = clf.fit(X_Train, y_Train)
score_int = accuracy_score( y_Test, clf.predict(X_Test))

if score_none >= score_int:
    print('the best max_depth is None')
    print(score_none)
else:
    print('the best max_depth is 10')
    print(score_int)

# visualize the pixel importance ################################################################
feature = numpy.reshape(clf.feature_importances_, [28, 28])
pylab.imshow(feature)
pylab.show()
