###############################################################################################
Problem Statement: A National Zoopark in India is dealing with the problem of segregation of the animals based on the different attributes they have. 
                   Build a KNN model to automatically classify the animals. 
                   Explain any inferences you draw in the documentation.

################################################################################################

# Importing required libraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer, BoxCoxTransformer


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Reading the dataset into Python
df = pd.read_csv("D:/Hands on/17_KNN/Assignments/Zoo.csv")

# Information of the dataset
df.info()

# Statistical calculations of the dataset
df.describe()

# Datatypes of the dataset
df.dtypes

# Returns top five records
df.head()

# First moment business decession
df.mean()

df.median()

df.mode()

# Second moment business decession
df.var()

df.std()

# Third moment business decession
df.skew()

# Fourth moment business decession
df.kurt()

# Correlation coefficient
df.corr()

# Pairplot
sns.pairplot(df)

# Checking the unique values of 'Animal name' feature
df['animal name'].value_counts()
df['animal name'].nunique()

# Deleting 'animal name' since it is nominal data
df.drop(["animal name"], axis = 1, inplace = True)

# Shape of the dataset
df.shape

# Checking for duplicated data
dup = df.duplicated()

# Number of duplicate values
dup.sum()

# Deleting the duplicates
df.drop_duplicates(inplace = True)

# Checking for duplicated data
df.duplicated().sum()

# Checking for null values
df.isnull().sum()
df.isna().sum()

# Spliting the dataset into X(input) and Y(output)
Y = df['type']
X = df.iloc[:, 0:16]

# Normalization on X dataset. min = 0, max = 1
Scale = StandardScaler()
X_norm = pd.DataFrame(Scale.fit_transform(X))

# Spliting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size = 0.2, random_state = 0)

# Creating a KneighborClassfier object
Kneighbor = KNeighborsClassifier(n_neighbors = 5)

# Building a model
Knnmodel = Kneighbor.fit(x_train, y_train)

# Predicting on train data
train_pred = Knnmodel.predict(x_train)

# Confusion matrix on trainind data
confusion_matrix(y_train, train_pred)

# Training accuracy
accuracy_score(y_train, train_pred)

# Predicting on testing data
test_pred = Knnmodel.predict(x_test)

# Confusion matix on testing data
confusion_matrix(y_test, test_pred)

# Accuracy on test data
accuracy_score(y_test, test_pred)

# Creating a dictionary with range of values
krange = np.arange(3, 50, 2)
param_grid = dict(n_neighbors = krange)

# Creating a GridSearchCV object
grid = GridSearchCV(Kneighbor, param_grid, scoring = 'accuracy', return_train_score = False, verbose = 1)

# Building a model
gridmodel = grid.fit(x_train, y_train)

# Best estimator
print(gridmodel.best_estimator_)

# Best params
print(gridmodel.best_params_)

# Best score
print(gridmodel.best_score_)

# Prediction on training data
train_pred1 = gridmodel.predict(x_train)

# Confusion matrix on train data
confusion_matrix(y_train, train_pred1)

# Accuracy on train data
accuracy_score(y_train, train_pred1)

# Prediction on testing data
test_pred1 = gridmodel.predict(x_test)

# Confusion table on testing data
cm= confusion_matrix(y_test, test_pred1)

# Accuracy on testing data
accuracy_score(y_test, test_pred1)

# Confusion table on testing data with labels
cmplot = ConfusionMatrixDisplay(confusion_matrix = cm)
cmplot.plot()
cmplot.ax_.set(title = 'Type prediction - Confusion matrix', xlabel = 'Predicted', ylabel = 'Actual')
