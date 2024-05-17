import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

# DATA IMPORTING AND PROCESSING

# TRAINING SET
data = "./32-bit/train_32bit.txt"
dfTrain1 = pd.read_csv(data, header=None)
dfTrain1 = dfTrain1.dropna()
data = "./64-bit/train_64bit.txt"
dfTrain2 = pd.read_csv(data, header=None)
dfTrain2 = dfTrain2.dropna()
data = "./128-bit/train_128bit.txt"
dfTrain3 = pd.read_csv(data, header=None, dtype={"a": "Int64", "b": "Int64"})
dfTrain3 = dfTrain3.dropna()

# VALIDATION SET
data = "./32-bit/val_32bit.txt"
dfVal1 = pd.read_csv(data, header=None)
dfVal1 = dfVal1.dropna()
data = "./64-bit/val_64bit.txt"
dfVal2 = pd.read_csv(data, header=None)
dfVal2 = dfVal2.dropna()
data = "./128-bit/val_128bit.txt"
dfVal3 = pd.read_csv(data, header=None, dtype={"a": "Int64", "b": "Int64"})
dfVal3 = dfVal3.dropna()

# separating validation set into data and labels before adding val set to training set
xVal1 = dfVal1.iloc[:, :-1]
yVal1 = dfVal1.iloc[:, -1]
xVal2 = dfVal2.iloc[:, :-1]
yVal2 = dfVal2.iloc[:, -1]
xVal3 = dfVal3.iloc[:, :-1]
yVal3 = dfVal3.iloc[:, -1]

# adding them together for non-validation-requiring ML models
dfNew1 = pd.concat([dfTrain1, dfVal1])
dfNew2 = pd.concat([dfTrain2, dfVal2])
dfNew3 = pd.concat([dfTrain3, dfVal3])
dfNew1 = dfNew1.dropna()
dfNew2 = dfNew2.dropna()
dfNew3 = dfNew3.dropna()

# separate data and labels again
xTrain1 = dfNew1.iloc[:, :-1]
yTrain1 = dfNew1.iloc[:, -1] 
xTrain2 = dfNew2.iloc[:, :-1]
yTrain2 = dfNew2.iloc[:, -1] 
xTrain3 = dfNew3.iloc[:, :-1] 
yTrain3 = dfNew3.iloc[:, -1] 

# also create a CNN-friendly training set with just the training data
xTrain1C = dfTrain1.iloc[:, :-1]
yTrain1C = dfTrain1.iloc[:, -1] 
xTrain2C = dfTrain2.iloc[:, :-1]
yTrain2C = dfTrain2.iloc[:, -1] 
xTrain3C = dfTrain3.iloc[:, :-1] 
yTrain3C = dfTrain3.iloc[:, -1] 

# TESTING SET
data = "./32-bit/test_32bit.txt"
dfTest1= pd.read_csv(data, header=None)
dfTest1 = dfTest1.dropna()

data = "./64-bit/test_64bit.txt"
dfTest2= pd.read_csv(data, header=None)
dfTest2 = dfTest2.dropna()

data = "./128-bit/test_128bit.txt"
dfTest3 = pd.read_csv(data, header=None, dtype={"a": "Int64", "b": "Int64"})
dfTest3 = dfTest3.dropna()

# and yet again separate data nad labels
xTest1 = dfTest1.iloc[:, :-1]
yTest1 = dfTest1.iloc[:, -1]
xTest2 = dfTest2.iloc[:, :-1]
yTest2 = dfTest2.iloc[:, -1]
xTest3 = dfTest3.iloc[:, :-1]
yTest3 = dfTest3.iloc[:, -1]

# MODEL SETUP AND TESTING TIME!

# KNN
'''
# Optimizing to find the best number of neighbors
accuracy = 0
for i in range(1,26):
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(xTrain1, yTrain1)
    model = knn.predict(xTest1)
    newAccuracy = accuracy_score(yTest1, model)
    if newAccuracy > accuracy:
        accuracy = newAccuracy
        print(str(i), " is the best!")
'''

knn = KNeighborsClassifier(n_neighbors=16) 

# 32-bit
knn.fit(xTrain1, yTrain1)
model = knn.predict(xTrain1)
accuracy = accuracy_score(yTrain1, model)
print("KNN Training Accuracy 32-bit:", accuracy)    

model = knn.predict(xTest1)
accuracy = accuracy_score(yTest1, model)
print("KNN Testing Accuracy 32-bit:", accuracy)    

# 64-bit
knn.fit(xTrain2, yTrain2)
model = knn.predict(xTrain2)
accuracy = accuracy_score(yTrain2, model)
print("KNN Training Accuracy 64-bit:", accuracy)    

model = knn.predict(xTest2)
accuracy = accuracy_score(yTest2, model)
print("KNN Testing Accuracy 64-bit:", accuracy)    

# 128-bit
knn.fit(xTrain3, yTrain3)
model = knn.predict(xTrain3)
accuracy = accuracy_score(yTrain3, model)
print("KNN Training Accuracy 128-bit:", accuracy)  

model = knn.predict(xTest3)
accuracy = accuracy_score(yTest3, model)
print("KNN Testing Accuracy 128-bit:", accuracy)    
print()


# SVM
svm = svm.SVC()

# hyperparemeter tuning from https://scikit-learn.org/stable/modules/grid_search.html
# but it didn't change accuracy and increased execution time so not doing it! It seemed that 'rbf' was the best kernel,
# but it increased execution time even more than it already is so I left it as the default version,
# as the accuracy score increase was not significant
#param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "kernel": ["linear", "poly", "rbf", "sigmoid"]}
#grid_search = GridSearchCV(estimator=lin, param_grid=param_grid, scoring="accuracy")
#svm = grid_search

# 32-bit
svm.fit(xTrain1, yTrain1)
model = svm.predict(xTrain1)
accuracy = accuracy_score(yTrain1, model)
print("SVM Training Accuracy 32-bit:", accuracy)

model = svm.predict(xTest1)
accuracy = accuracy_score(yTest1, model)
print("SVM Testing Accuracy 32-bit:", accuracy)

# 64-bit
svm.fit(xTrain2, yTrain2)
model = svm.predict(xTrain2)
accuracy = accuracy_score(yTrain2, model)
print("SVM Training Accuracy 64-bit:", accuracy)

model = svm.predict(xTest2)
accuracy = accuracy_score(yTest2, model)
print("SVM Testing Accuracy 64-bit:", accuracy)

# 128-bit
svm.fit(xTrain3, yTrain3)
model = svm.predict(xTrain3)
accuracy = accuracy_score(yTrain3, model)
print("SVM Training Accuracy 128-bit:", accuracy)

model = svm.predict(xTest3)
accuracy = accuracy_score(yTest3, model)
print("SVM Testing Accuracy 128-bit:", accuracy)
print()


# Decision Tree setup
dtc = DecisionTreeClassifier()

# parameter tuning, from above source and looking at DT parameters
param_grid = {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 10, 15]}
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, scoring="accuracy")
dtc = grid_search

# 32-bit
dtc.fit(xTrain1, yTrain1)
model = dtc.predict(xTrain1)
accuracy = accuracy_score(yTrain1, model)
print("DT Training Accuracy 32-bit:", accuracy)

model = dtc.predict(xTest1)
accuracy = accuracy_score(yTest1, model)
print("DT Testing Accuracy 32-bit:", accuracy)

# 64-bit
dtc.fit(xTrain2, yTrain2)
model = dtc.predict(xTrain2)
accuracy = accuracy_score(yTrain2, model)
print("DT Training Accuracy 64-bit:", accuracy)

model = dtc.predict(xTest2)
accuracy = accuracy_score(yTest2, model)
print("DT Testing Accuracy 64-bit:", accuracy)

# 128-bit
dtc.fit(xTrain3, yTrain3)
model = dtc.predict(xTrain3)
accuracy = accuracy_score(yTrain3, model)
print("DT Training Accuracy 128-bit:", accuracy)

model = dtc.predict(xTest3)
accuracy = accuracy_score(yTest3, model)
print("DT Testing Accuracy 128-bit:", accuracy)
print()

# Logistic Regression setup
# base information from https://scikit-learn.org/stable/modules/linear_model.html
lin = linear_model.LogisticRegression(max_iter=1000)

# hyperparemeter tuning from above source
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(estimator=lin, param_grid=param_grid, scoring="accuracy")
lin = grid_search

# 32-bit
lin.fit(xTrain1, yTrain1)
model = lin.predict_proba(xTrain1)
accuracy = log_loss(yTrain1, model)
print("Logistic Regression Training Accuracy 32-bit:", accuracy)

model = lin.predict_proba(xTest1)
accuracy = log_loss(yTest1, model)
print("Logistic Regression Testing Accuracy 32-bit:", accuracy)    

# 64-bit
lin.fit(xTrain2, yTrain2)
model = lin.predict_proba(xTrain2)
accuracy = log_loss(yTrain2, model)
print("Logistic Regression Training Accuracy 64-bit:", accuracy)

model = lin.predict_proba(xTest2)
accuracy = log_loss(yTest2, model)
print("Logistic Regression Testing Accuracy 64-bit:", accuracy)    

# 128-bit
lin.fit(xTrain3, yTrain3)
model = lin.predict_proba(xTrain3)
accuracy = log_loss(yTrain3, model)
print("Logistic Regression Training Accuracy 128-bit:", accuracy)

model = lin.predict_proba(xTest3)
accuracy = log_loss(yTest3, model)
print("Logistic Regression Testing Accuracy 128-bit:", accuracy)    
print()