#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:43:41 2017

@author: ashish
"""
from pandas import  DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

#Load the dataset
data = pd.read_csv("gapminder.csv")

for column in data.columns:
    if data[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Data Management
data_clean = data.dropna()

#select predictor variables and target variable as separate data sets 
cluster = data_clean[['incomeperperson','alcconsumption','armedforcesrate','co2emissions','femaleemployrate','hivrate',
'internetuserate','oilperperson','polityscore','relectricperperson','suicideper100th','employrate','urbanrate']]


# standardize predictors to have mean=0 and sd=1
clustervar=cluster.copy()
from sklearn import preprocessing
clustervar['incomeperperson']=preprocessing.scale(clustervar['incomeperperson'].astype('float64'))
clustervar['alcconsumption']=preprocessing.scale(clustervar['alcconsumption'].astype('float64'))
clustervar['armedforcesrate']=preprocessing.scale(clustervar['armedforcesrate'].astype('float64'))
clustervar['co2emissions']=preprocessing.scale(clustervar['co2emissions'].astype('float64'))
clustervar['femaleemployrate']=preprocessing.scale(clustervar['femaleemployrate'].astype('float64'))
clustervar['hivrate']=preprocessing.scale(clustervar['hivrate'].astype('float64'))
clustervar['internetuserate']=preprocessing.scale(clustervar['internetuserate'].astype('float64'))
clustervar['oilperperson']=preprocessing.scale(clustervar['oilperperson'].astype('float64'))
clustervar['polityscore']=preprocessing.scale(clustervar['polityscore'].astype('float64'))
clustervar['relectricperperson']=preprocessing.scale(clustervar['relectricperperson'].astype('float64'))
clustervar['suicideper100th']=preprocessing.scale(clustervar['suicideper100th'].astype('float64'))
clustervar['employrate']=preprocessing.scale(clustervar['employrate'].astype('float64'))
clustervar['urbanrate']=preprocessing.scale(clustervar['urbanrate'].astype('float64'))

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')


# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()



"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
xyz=[0,1,2]
clus_train.reset_index(level = xyz, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))

# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
#newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
xyz=[0,1,2]
newclus.reset_index(level= xyz, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


# validate clusters in training data by examining cluster differences in lifeexpectancy using ANOVA
# first have to merge lifeexpectancy with clustering variables and cluster assignment data 
life_data=data_clean['lifeexpectancy']
# split lifeexpectancy data into train and test sets
life_train, gpa_test = train_test_split(life_data, test_size=.3, random_state=123)
life_train1=pd.DataFrame(life_train)

life_train1.reset_index(level=xyz, inplace=True)
merged_train_all=pd.merge(life_train1, merged_train, on='index')
sub1 = merged_train_all[['lifeexpectancy', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

gpamod = smf.ols(formula='lifeexpectancy ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())

print ('means for lifeexpectancy by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for lifeexpectancy by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['lifeexpectancy'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())