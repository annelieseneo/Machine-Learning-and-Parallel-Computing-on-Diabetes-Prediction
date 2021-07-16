# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:16:15 2021

@author: User
"""

# import pandas
import pandas as pd

# import numpy for working with numbers
import numpy as np

# import plots
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')


# DATA SELECTION/COLLECTION

# import data from Excel csv sheet
df = pd.read_csv('C:\\Users\\User\\Downloads\\ITS66604-MLPC\\diabetes_data.csv')

# show first 5 records of dataset
df.head()

# determine object type of dataset
type(df)


# HIGH LEVEL STATISTICS

# summary statistics of the attributes, including measures of central tendency
# and measures of dispersion
df.describe()

# boxplot
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), return_type = 'axes', figsize = (30, 8))
plt.title("Box Plot of Diabetes Data") # title of plot
plt.suptitle("")
plt.xlabel("Attribute") # x axis label
plt.ylabel("Measurements (units)") # y axis label
plt.show()

# import searborn library for more variety of data visualisation using 
# fewer syntax and interesting default themes
import seaborn as sns 

# compare linear relationships between attribtues using correlation coefficient
# generated using correlation matrix
sns.heatmap(df.corr(), cmap = 'PuBu', annot = True)
plt.show()

# visualise pairs plot or scatterplot matrix in relation to diabetes outcome
g = sns.pairplot(df, hue = 'Outcome', palette = 'PuBu')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)

# pie chart of count of target class label 'Outcome'
labels = 'Positive','Negative'
from collections import Counter
count = Counter(df['Outcome'])
sizes = [str(count[1]),str(count[0])]
colors = ['tomato','royalblue']
explode = (0,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)


# DATA PREPROCESSING

# data cleaning: missing values
# display the number of entries, the number of names of the column attributes,
# the data type and digit placings, and the memory space used
df.info()

# data cleaning: noises of impossible values
# identify impossible values and outliers using boxplot
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), return_type = 'axes', figsize = (30, 8))
plt.title("Box Plot of Diabetes Data") # title of plot
plt.suptitle("")
plt.xlabel("Attribute") # x axis label
plt.ylabel("Measurements (units)") # y axis label
plt.show()

# summary statistics of the attributes, including measures of central tendency
# and measures of dispersion
df.describe()

# smooth impossible values by replacing the value with the mean value
df['Glucose'] = df['Glucose'].replace(0, df.Glucose.mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df.BloodPressure.mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df.SkinThickness.mean())
df['Insulin'] = df['Insulin'].replace(0, df.Insulin.mean())
df['BMI'] = df['BMI'].replace(0, df.BMI.mean())

# data cleaning: anomalous outliers
# smooth outliers using winsorization technique
# replace outlier with maximum or minimum non-outlier

# attributes
fn = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
      'BMI','DiabetesPedigreeFunction','Age']

for feature in fn:
    #compute interquartile range (IQR)
    IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)
    
    # compute maximum and minimum non-outlier value
    minAllowed = df[feature].quantile(0.25)-1.5*IQR
    maxAllowed = df[feature].quantile(0.75)+1.5*IQR
    
    # replace outlier values
    for i in range (len(df[feature])):
        if df[feature][i] < minAllowed:
            df[feature] = df[feature].replace(df[feature][i], minAllowed)
        elif df[feature][i] > maxAllowed:
            df[feature] = df[feature].replace(df[feature][i], maxAllowed)
        else: continue
    
# confirmed smoothed outliers an dimpossible values using boxplot
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), return_type = 'axes', figsize = (30, 8))
plt.title("Box Plot of Diabetes Data") # title of plot
plt.suptitle("")
plt.xlabel("Attribute") # x axis label
plt.ylabel("Measurements (units)") # y axis label
plt.show()

# summary statistics of the attributes, including measures of central tendency and 
# measures of dispersion
df.describe()

# data cleaning: detect duplicated records
df[df.duplicated(subset = None, keep = False)]

# data transformation: Min Max Scaling
from sklearn.preprocessing import MinMaxScaler

# define scaler
scaler = MinMaxScaler(feature_range=(0,1))

# transform the data
scale_df = scaler.fit_transform(df)
names=df.columns
scaled_df = pd.DataFrame(scale_df, columns=names)

# data reduction: correlation matrix
# compare linear relationships between attributes using correlation coefficient generated using
# correlation matrix
sns.heatmap(scaled_df.corr(), cmap = 'PuBu', annot = True)
plt.show()

# visualise pairs plot or scatterplot matrix to identify weak class-attribute relationship
g = sns.pairplot(scaled_df, hue = 'Outcome', palette = 'PuBu')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)

# display the number of entries, the number and names of the column attributes, the data type and
# digit placings, and the memory space used
scaled_df.info()


# EXPLORATORY DATA ANALYSIS AND DATA VISUALIZATION

# list and count the target class label names and their frequency
from collections import Counter
count = Counter(scaled_df['Outcome'])
count.items()

# count of each target class label, and plot a column chart
plt.figure(figsize = (5, 5))
ax = sns.countplot(df['Outcome'], palette = 'PuBu')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, ha = "right")
plt.suptitle("Count of Diabetes Outcome")
plt.show()

# compare linear relationships between attributes using correlation coefficient generated using
# correlation matrix
sns.heatmap(scaled_df.corr(), cmap = 'PuBu', annot = True)
plt.show()

# visualise pairs plot or scatterplot matrix to identify weak class-attribute relationship
g = sns.pairplot(scaled_df, hue = 'Outcome', palette = 'PuBu')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)

# summarise main characteristics by displaying the summary statistics of the attributes, including 
# measures of central tendency, and measures of dispersion
scaled_df.describe()


# DATA MINING AND MODELLING

# classify and model the data using Decision Tree (DT), k-Neaarest Neighbours (KNN),
# and Gaussian Naive Bayes (NB) machine learning algorithms

# import train test split module
from sklearn.model_selection import train_test_split

# import DT algorithm from DT class
from sklearn.tree import DecisionTreeClassifier

# import KNN algorithm from KNN class
from sklearn.neighbors import KNeighborsClassifier

# import Gaussian NB algorithm form NB class
from sklearn.naive_bayes import GaussianNB
import math

# split dataset into attributes and labels
X = scaled_df.iloc[:, :-1].values # the attributes
y = scaled_df.iloc[:, 8].values # the labels

# choose appropriate range of training set proportions
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# plot decision tree based on entropy information gain, best splitter, and minimum 2
# sample leaves
DT = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy', min_samples_leaf = 2)

# use Gaussian method to support continuous data values
NB = GaussianNB()

# choose recommended optimal number of clusters of sqrt(number of records)
KNN = KNeighborsClassifier(n_neighbors = math.ceil(math.sqrt(768)))

# find best training set proportion for the chosen models
plt.figure()
for s in t:
    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s, random_state = 19)
        DT.fit(X_train, y_train) # consider DT scores
        scores.append(DT.score(X_test, y_test))
        NB.fit(X_train, y_train) # consider NB scores
        scores.append(NB.score(X_test, y_test))
        KNN.fit(X_train, y_train) # consider KNN scores
        scores.append(KNN.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training Set Proportion') # x axis label
plt.ylabel('Accuracy'); # y axis label

# choose train test splits from original dataset as 70% train data and 30% test data for highest accuracy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19)

# find optimal k number of clusters
k_range = range(1, 25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k') # x axis label
plt.ylabel('Accuracy') # y axis label
plt.scatter(k_range, scores) # scatter plot
plt.xticks([0, 5, 10, 15, 20, 25]);

# number of records in training set
len(X_train)

# count each outcome in training set
count = Counter(y_train)
print(count.items())

# fit KNN classifier
# choose 19 as the optimal number of clusters
classifierKNN = KNeighborsClassifier(n_neighbors = 19)
classifierKNN.fit(X_train, y_train)

# most effective KNN distance metric
classifierKNN.effective_metric_

# fit NB classifier
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)

# show prior probability of each class
classifierNB.class_prior_

# using DT classifier based on entropy information gain, best splitter, and minimum 2
# sample leaves
classifierDT = DecisionTreeClassifier(splitter = 'best', criterion='entropy', 
                                      min_samples_leaf = 2)
classifierDT.fit(X_train, y_train)

# plot decision tree
from sklearn import tree
fig = plt.figure(figsize = (100, 70))
cn = ['1','0']
DT = tree.plot_tree(classifierDT,
                    feature_names = fn,  
                    class_names = cn,
                    filled = True)

# identifies the important features
classifierDT.feature_importances_

# extracted rules
dtrules = tree.export_text(classifierDT, feature_names = fn)
print(dtrules)


# MODEL EVALUATION AND VALIDATION

# number of records in test set
len(X_test)

# count each outcome in test set
count = Counter(y_test)
print(count.items())

# use the chosen three models to make predictions on test data
y_predKNN = classifierKNN.predict(X_test)
y_predDT = classifierDT.predict(X_test)
y_predNB = classifierNB.predict(X_test)

# for KNN model
# using confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predKNN))
print(classification_report(y_test, y_predKNN))

# using accuracy performance metric
from sklearn.metrics import accuracy_score
print("Train Accuracy: ", accuracy_score(y_train, classifierKNN.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predKNN))


# for NB model
# using confusion matrix
print(confusion_matrix(y_test, y_predNB))
print(classification_report(y_test, y_predNB))

# using accuracy performance metric
print("Train Accuracy: ", accuracy_score(y_train, classifierNB.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predNB))


# for DT model
# using confusion matrix
print(confusion_matrix(y_test, y_predDT))
print(classification_report(y_test, y_predDT))

# using accuracy performance metric
print("Train Accuracy: ", accuracy_score(y_train, classifierDT.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predDT))


# performance data to plot
n_groups = 3
algorithms = ('k-Nearest Neighbour (KNN)', 
              'Decision Tree (DT)', 
              'Naive Bayes (NB)')
train_accuracy = (accuracy_score(y_train, classifierKNN.predict(X_train))*100, 
                  accuracy_score(y_train, classifierDT.predict(X_train))*100, 
                  accuracy_score(y_train, classifierNB.predict(X_train))*100)
test_accuracy = (accuracy_score(y_test, y_predKNN)*100, 
                 accuracy_score(y_test, y_predDT)*100, 
                 accuracy_score(y_test, y_predNB)*100)

# create plot to determine the best model
fig, ax = plt.subplots(figsize=(15, 5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
rects1 = plt.bar(index, train_accuracy, bar_width, alpha = opacity, 
                 color='Cornflowerblue', label='Train')
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width, alpha = opacity, 
                 color='Teal', label='Test')
plt.xlabel('Algorithm') # x axis label
plt.ylabel('Accuracy (%)') # y axis label
plt.ylim(0, 115)
plt.title('Comparison of Algorithm Accuracies') # plot title
plt.xticks(index + bar_width * 0.5, algorithms) # x axis data labels
plt.legend(loc = 'upper right') # show legend
for index, data in enumerate(train_accuracy):   
    plt.text(x = index - 0.035, y = data + 1, s = round(data, 2), 
             fontdict = dict(fontsize = 8))
for index, data in enumerate(test_accuracy):
    plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), 
             fontdict = dict(fontsize = 8))
plt.show()


# MODEL INTERPRETATION

# new data record must be within the data ranges to avoid extrapolation
scaled_df.describe()

# create new record
newdata = [[1, 50, 80, 33, 70, 30, 0.55, 25]]

# transform the new data
scale_newdf = scaler.fit_transform(newdata)
scaled_newdf = pd.DataFrame(scale_newdf, columns=fn)

# compute probabilities of assigning to each of the two classes of Outcome
probaKNN = classifierKNN.predict_proba(scaled_newdf)
probaKNN.round(4) # round probabilities to four decimal places, if applicable

# make prediction of class label
predKNN = classifierKNN.predict(scaled_newdf)
predKNN













