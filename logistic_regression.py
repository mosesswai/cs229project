import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

##Some configuration settings
pd.set_option("display.max_columns", 100)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#import the data
df = pd.read_csv('african_crises.csv', index_col = 0) #index col neccessary?
df.head()

print(df.shape) #(1059, 13)


# Plot total number of cases by country
sns.countplot(x='country', data=df, palette ='hls')
plt.xticks(rotation=90)
plt.show()

# Plot counts of 'x'_crisis, where 'x' can equals: systemic, currency, inflation, branking.
sns.countplot(x='systemic_crisis', data=df, palette ='hls')
plt.xticks(rotation=90)
plt.show()
print(df['banking_crisis'].value_counts()) #print counts

#------------------------------------------------------------------------------------------
#						         	PRE-PROCESSING
#------------------------------------------------------------------------------------------

# Drop the categorical variables which are not useful to the dataset for logistic regression
df = df.drop(['cc3', 'country', 'year'], axis =1)
df.head()

# drop this column since it is not informative (jsutify with plot later)
df = df.drop(['gdp_weighted_default'], axis =1)
df.head()

# Define ouput column y and drop from dataset
y = df[['banking_crisis']]
y = pd.get_dummies(df['banking_crisis'],drop_first=True)
df = df.drop(['banking_crisis'], axis =1)
df.head()


#------------------------------------------------------------------------------------------
#						         	  TRAINING
#------------------------------------------------------------------------------------------

# split the data into test train sets
from sklearn.model_selection import train_test_split
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# fitting
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
#predictions
Predictions = logmodel.predict(X_test)


#------------------------------------------------------------------------------------------
#						         	  REPORT
#------------------------------------------------------------------------------------------
from sklearn.metrics import classification_report
print(classification_report(y_test,Predictions))
# confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, Predictions))

# calculate the fpr and tpr for all thresholds of the classification
import sklearn.metrics as metrics
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)



#------------------------------------------------------------------------------------------
#						         	  DISPLAY
#------------------------------------------------------------------------------------------


# method I: plt
import matplotlib.pyplot as plt
plt.title('Flase Positives /True Positives')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
