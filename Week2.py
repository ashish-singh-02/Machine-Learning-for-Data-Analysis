# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:27:31 2017

@author: Ashish
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

#Load the dataset
GM_data = pd.read_csv("gapminder.csv")

from sklearn import preprocessing
for column in GM_data.columns:
    if GM_data[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        GM_data[column] = le.fit_transform(GM_data[column])

data_clean = GM_data.dropna()

data_clean.dtypes
print(data_clean.describe())

#Split into training and testing sets
predictors = data_clean[['incomeperperson','alcconsumption','armedforcesrate','co2emissions','femaleemployrate','hivrate',
'internetuserate','oilperperson','polityscore','relectricperperson','suicideper100th','employrate','urbanrate']]

targets = data_clean.lifeexpectancy


pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape


#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

my_conf_matrix = sklearn.metrics.confusion_matrix(tar_test, predictions)
print('Confusion matrix')
print (my_conf_matrix)
print('\n'*2)
my_accurate_score = sklearn.metrics.accuracy_score(tar_test, predictions)
print('Accuracy score')
print(my_accurate_score)


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute

#To print the data of complete array
np.set_printoptions(threshold=np.inf)
print('\n'*2)
print('Relative Importance of Each Variable :')
print(model.feature_importances_)

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)

