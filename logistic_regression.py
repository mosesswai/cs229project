import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

#------------------------------------------------------------------------------------------
#				  CONFIGURATION
#------------------------------------------------------------------------------------------

# Dataset pre processing configuration
remove_kaggle_parameters = False
include_extra_parameters = True

##Some configuration settings
pd.set_option("display.max_columns", 100)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#------------------------------------------------------------------------------------------
#				  IMPORT DATA
#------------------------------------------------------------------------------------------

# Import the main data
dataframe = pd.read_csv('data/african_crises.csv')
print(dataframe.shape) #(1059, 13)

# Add a unique index to each data point
dataframe['idx'] = np.arange(len(dataframe))
dataframe.set_index('idx', inplace=True)


# Add extra parameter from a .csv file to the original dataframe
def add_parameter(dataframe, param_name, file_name):
    param_data = pd.read_csv(file_name, index_col=1)

    # Add new parameter column
    dataframe[param_name] = np.nan

    for cc in param_data.index:
        # Extract data batch based on country code from original dataframe
        # and make the year the index
        temp_data = dataframe.loc[dataframe['cc3'] == cc]
        temp_data = temp_data.reset_index()
        temp_data = temp_data.set_index('year')
        
        # Extract data batch based on country code from parameter data
        temp_param_data = param_data.loc[[cc]].transpose().iloc[2:]
        temp_param_data.columns = [param_name]
        temp_param_data.index = temp_param_data.index.astype('int64') 

        # Update the temporary data batch to add parameter data then
        # update the original dataframe
        temp_data.update(temp_param_data)
        temp_data = temp_data.reset_index()
        temp_data = temp_data.set_index('idx')
        dataframe.update(temp_data)

# Include extra parameters if configured
if include_extra_parameters:
    #G1
    # add_parameter(dataframe, 'population_growth','data/data_population_growth.csv')
    # add_parameter(dataframe, 'foreign_aid','data/data_foreign_aid.csv')
    #G2
    # add_parameter(dataframe, 'gdp_growth','data/data_gdp_growth.csv')
    # add_parameter(dataframe, 'gni','data/data_gni.csv')
    #G3
    # add_parameter(dataframe, 'fdi','data/data_fdi.csv')
    # add_parameter(dataframe, 'ext_debt','data/data_ext_debt_stocks.csv')
    #G4
    # add_parameter(dataframe, 'unemployment','data/data_unemployment.csv')

    dataframe.dropna(inplace = True)
    print(dataframe.shape) 

# Plot total number of cases by country
sns.countplot(x='country', data=dataframe, palette ='hls')
plt.xticks(rotation=90)
# plt.show()

# Plot counts of 'x'_crisis, where 'x' can equals: systemic, currency, inflation, branking.
sns.countplot(x='systemic_crisis', data=dataframe, palette ='hls')
plt.xticks(rotation=90)
# plt.show()
print(dataframe['banking_crisis'].value_counts()) #print counts

#------------------------------------------------------------------------------------------
#				  PRE-PROCESSING
#------------------------------------------------------------------------------------------

# Drop the categorical variables which are not useful to the dataset for logistic regression
dataframe.reset_index(inplace = True)
dataframe = dataframe.drop(['idx', 'cc3', 'country', 'year'], axis =1)
dataframe.head()

# drop this column since it is not informative (jsutify with plot later)
dataframe = dataframe.drop(['gdp_weighted_default'], axis =1)
dataframe.head()

# Define ouput column y and drop from dataset
y = dataframe[['banking_crisis']]
y = pd.get_dummies(dataframe['banking_crisis'],drop_first=True)
dataframe = dataframe.drop(['banking_crisis'], axis =1)
dataframe.head()


#------------------------------------------------------------------------------------------
#				  TRAINING
#------------------------------------------------------------------------------------------

# split the data into test train sets
from sklearn.model_selection import train_test_split
# create training and testing vars
X_train, X_test, Y_train, Y_test = train_test_split(dataframe, y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
# train
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
# predict
Predictions = logmodel.predict(X_test)


#------------------------------------------------------------------------------------------
#			      REPORT
#------------------------------------------------------------------------------------------
from sklearn.metrics import classification_report
print(classification_report(Y_test,Predictions))
# confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Predictions))

# calculate the fpr and tpr for all thresholds of the classification
import sklearn.metrics as metrics
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)



#------------------------------------------------------------------------------------------
#			      DISPLAY
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
# plt.show()
