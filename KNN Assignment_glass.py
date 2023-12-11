###############################################################################
Problem Statement : A glass manufacturing plant uses different earth elements to design new glass materials based on customer requirements. 
                    For that, they would like to automate the process of classification as it’s a tedious job to manually classify them. 
                    Help the company achieve its objective by correctly classifying the glass type based on the other features using KNN algorithm.
###############################################################################
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sma

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from feature_engine.outliers import Winsorizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the dataset into Python
data = pd.read_csv("D:/Hands on/17_KNN/Assignments/glass.csv")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Pairplot
sns.pairplot(data)

# Correlation coefficient
data.corr()

# Checking for duplicated data
dup = data.duplicated()

# Number of duplicate values
dup.sum()

# Droping the duplicate values
data.drop_duplicates(inplace = True)

# Checking for duplicated data
dup = data.duplicated()

# Number of duplicate values
dup.sum()

# Checking for null values
data.isna().sum()
data.isnull().sum()

# Spliting the dataset into X(input) and Y(output)
X = data.iloc[:, 0:9]

Y = data.Type

# Checking for unique values and its count
Y.value_counts()

# for loop to draw boxplot on X dataset columns
for i in X.columns:
    sns.boxplot(X[i])
    plt.title('Box plot for ' + str(i))
    plt.show()
    

# Applying winsorizer on features which are having outliers
RI_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['RI'])
X['RI'] = RI_winsor.fit_transform(X[['RI']])

Na_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Na'])
X['Na'] = Na_winsor.fit_transform(X[['Na']])

Al_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Al'])
X['Al'] = Al_winsor.fit_transform(X[['Al']])

Si_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Si'])
X['Si'] = Si_winsor.fit_transform(X[['Si']])

K_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['K'])
X['K'] = K_winsor.fit_transform(X[['K']])

Ca_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Ca'])
X['Ca'] = Ca_winsor.fit_transform(X[['Ca']])

Ba_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Ba'])
X['Ba'] = Ba_winsor.fit_transform(X[['Ba']])

Fe_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Fe'])
X['Fe'] = Fe_winsor.fit_transform(X[['Fe']])

# for loop to draw boxplot after Winsorization is done
for i in X.columns:
    sns.boxplot(X[i])
    plt.title('Box plot for ' + str(i))
    plt.show()

# Normalization on X dataset. min = 0, max = 1
minmax = MinMaxScaler()
X_scale = pd.DataFrame(minmax.fit_transform(X))

# Spliting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X_scale, Y, test_size = 0.2, random_state = 0)

# Creating a KneighborClassfier object
Kneighbor = KNeighborsClassifier(n_neighbors = 11)

# Building a model
model = Kneighbor.fit(x_train, y_train)

# Predicting on train data
train_pred = model.predict(x_train)

# Cross table
pd.crosstab(y_train, train_pred)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Training accuracy
accuracy_score(y_train, train_pred)

# Predicting on testing data
test_pred = model.predict(x_test)

# Cross table
pd.crosstab(y_test, test_pred)

# Confusion matix
confusion_matrix(y_test, test_pred)

# Accuracy on test data
accuracy_score(y_test, test_pred)

# Tryind different n_neighbor values to see which is giving better accuracy
k = []
l = []
acc = []
for i in range(3, 50, 2):
    Kneighbor = KNeighborsClassifier(n_neighbors = i)
    model = Kneighbor.fit(x_train, y_train)
    
    train_pred = accuracy_score(y_train, model.predict(x_train))
    test_pred = accuracy_score(y_test, model.predict(x_test))
    accu_diff = train_pred- test_pred
    
    acc.append([accu_diff, train_pred, test_pred])

acc

# plot to see where training and testing accuracy are high and close to each other 
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], 'ro-')
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], 'ro-')


# Creating a dictionary with range of values
krange = range(3, 50, 2)
param_grid = dict(n_neighbors = krange)  

Knn = KNeighborsClassifier()  

# Creating a GridSearchCV object
grid = GridSearchCV(Knn, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)    

# Building a model
knn_grid = grid.fit(x_train, y_train)    

# Best params
print(knn_grid.best_params_)

# Best estimator
print(knn_grid.best_estimator_)

# Best score
print(knn_grid.best_score_)

# Prediction on training data
train_pred1 = knn_grid.predict(x_train)

# Cross table on train data
pd.crosstab(y_train, train_pred1)

# Confusion matrix on train data
confusion_matrix(y_train, train_pred1)

# Accuracy on train data
accuracy_score(y_train, train_pred1)

# Prediction on testing data
test_pred1 = knn_grid.predict(x_test)

# Cross table on testing data
pd.crosstab(y_test, test_pred1)

# Confusion table on testing data
confusion_matrix(y_test, test_pred1)

# Accuracy on testing data
accuracy_score(y_test, test_pred1)