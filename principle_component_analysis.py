import os
import arff
from sklearn.cross_validation import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import itertools
#from confusion_matrix import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition
import matplotlib
import json




'''
Parameters
'''

# General
number_of_training_items = 2 # Choose the number of training cases (to limit the training time)


path_files = '/Users/eelcowiechert/dcase2016/data' # Where the weka files are

'''
Start Script
'''

print('Loading the file names')
t1=time.time()
files = []
for filename in os.listdir(path_files):
    files.append(filename)
t2=time.time()
t_past = t2-t1
print('Done loading files in %s seconds' % t_past)

print('There are %s files loaded' % len(files))

print('Loading the Data')

t1=time.time()

x = []
y = []
count=0
for file in files:
    count+=1
    if count >number_of_training_items:
        break
    print( '%s of %s' % (count, len(files)))
    for file in arff.load(open(path_files + '/' + file, mode="r"))["data"]: #for every row in file
        temp_list = []
        for number in file:
            temp_list.append(number)
        x.append(temp_list[:-1]) # data
        y.append(temp_list[-1]) # labels

t2=time.time()
t_past = t2-t1

labels = np.unique(y)

print('Done loading files in %s seconds' % t_past)

'''
Performing PCA
'''
x_variables = []
y_variables = []


for number in range(60):
    pca = decomposition.PCA(n_components=(number+1))
    pca.fit(x)
    x_temp = pca.transform(x)
    x_variables.append(number+1)
    y_variables.append(np.sum(pca.explained_variance_ratio_))

x2_variables = []
for number in pca.explained_variance_ratio_:
    x2_variables.append(number)

print x_variables
print x2_variables
print y_variables

'''
Print plot
'''

plt.plot(x_variables, y_variables, 'o-', color='black')
plt.bar(np.arange(60)+1, x2_variables, align='center', color='blue', alpha=0.5)
plt.title('Cumulative variance plot')
plt.xlabel('Principle component')
plt.ylabel('Cumulative proportion of variance')
plt.xlim([0,60])
plt.ylim([0,1.01])
plt.savefig('cum_var_plot.pdf', format='pdf', dpi=300)
plt.close()
