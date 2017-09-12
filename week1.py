# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:28:03 2017

@author: Ashish
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn.metrics


"""
Data Engineering and Analysis
"""
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

"""
Modeling and Prediction
"""
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
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

my_conf_matrix = sklearn.metrics.confusion_matrix(tar_test, predictions)
print('Confusion matrix')
print (my_conf_matrix)

my_accurate_score = sklearn.metrics.accuracy_score(tar_test, predictions)
print('Accuracy score')
print(my_accurate_score)


#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png ("Decision_Tree_Image.png")
