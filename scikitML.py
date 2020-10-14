import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


from sklearn import preprocessing 
from sklearn.decomposition import PCA 
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from scipy import interp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import operator


get_ipython().run_line_magic('matplotlib', 'inline')


# # Logistic Regression

train_data = pd.read_csv("train_2v.csv")

print(train_data)

train_data['smoking_status'] = train_data['smoking_status'].replace(np.nan, 'No Info')

train_data['bmi'] = train_data['bmi'].fillna(train_data['bmi'].mean())


# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
train_data['gender']= le.fit_transform(train_data['gender']) 
train_data['ever_married']= le.fit_transform(train_data['ever_married'])
train_data['work_type']= le.fit_transform(train_data['work_type']) 
train_data['Residence_type']= le.fit_transform(train_data['Residence_type'])
train_data['smoking_status']= le.fit_transform(train_data['smoking_status'])


X = train_data.loc[:, train_data.columns!='label']
y = train_data.loc[:, train_data.columns=='label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


from time import *
start_time = time()

# train_data['label'].value_counts()
train_data.groupby(['label']).agg(['count'])

end_time = time()
elapsed_time = end_time - start_time
print("time for operation: %.3f seconds" % elapsed_time)



from time import *
start_time = time()

clf = LogisticRegression()
clf.fit(X_train, y_train)

# likely take a fair amount of time

end_time = time()
elapsed_time = end_time - start_time
print("time to train model: %.3f seconds" % elapsed_time)


# Predicting the Test set results
y_pred = clf.predict(X_test)


print(list(y_test['label']))
print(y_pred)


# print(y_pred)
print(X_test.shape)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

print(classification_report(y_true = y_test['label'], y_pred = y_pred))



from time import *
start_time = time()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(clf, hyperparameters, cv=5, verbose=0)


# Fit grid search
best_model = clf.fit(X_train, y_train)

# likely take a fair amount of time

end_time = time()

elapsed_time = end_time - start_time

print("time to train model: %.3f seconds" % elapsed_time)



# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# Predict target vector
best_model.predict(X_test)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 5, random_state = 0)

from time import *
start_time = time()

clf.fit(X_train, y_train)


# likely take a fair amount of time

end_time = time()
elapsed_time = end_time - start_time
print("time to train model: %.3f seconds" % elapsed_time)



# Predict for 1 observation
clf.predict(X_test.iloc[0].values.reshape(1, -1))
# Predict for multiple observations
clf.predict(X_test)

# the score method returns the accuracy of the model
score = clf.score(X_test, y_test)
print(score)
