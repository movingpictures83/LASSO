#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[328]:


################################################ Preprocessing #########################################################
class LASSOPlugin:
 def input(self, inputfile):
  self.data_path = inputfile
 def run(self):
  pass
 def output(self, outputfile):
  #categorical_cols = ["Race_Ethnicity"]
  
  
  data_df = pd.read_csv(self.data_path)

  # # Tramsform categorical data to categorical format:
  # for category in categorical_cols:
  #     data_df[category] = data_df[category].astype('category')
  #

  # Clean numbers:
  #"Cocain_Use": {"yes":1, "no":0},
  cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
  }

  data_df.replace(cleanup_nums, inplace=True)

  # Drop id column:
  data_df = data_df.drop(["pilotpid"], axis=1)

  # remove NaN:
  data_df = data_df.fillna(0)

  # Standartize variables
  from sklearn import preprocessing
  names = data_df.columns
  scaler = preprocessing.StandardScaler()
  data_df_scaled = scaler.fit_transform(data_df)
  data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)


  # In[303]:


  ################################################ Users vs Non-Users #########################################################

  # Random Forest
  # Benchmark

  y_col = "Cocain_Use"
  test_size = 0.3
  validate = True

  y = data_df[y_col]

  X = data_df_scaled.drop([y_col], axis=1)

  # Create random variable for benchmarking
  X["random"] = np.random.random(size= len(X))

  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = test_size, random_state = 2)



  # In[305]:


  # Lasso Feature selection

  from sklearn.linear_model import LassoCV
  from sklearn.feature_selection import SelectFromModel
  from sklearn.linear_model import LogisticRegression

  #clf = LassoCV(cv=5)


  sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2', random_state=41)) # Lasso L1 penalty
  sel_.fit(X_train, y_train)

  selected_feat = list(X_train.columns[(sel_.get_support())])
  print(selected_feat)

  X_lasso = X[selected_feat]


  # In[233]:


  # Retrain the model with random forest with selected features
  X_train_lasso, X_valid_lasso, y_train_lasso, y_valid_lasso = train_test_split(X_lasso, y, test_size = test_size, random_state = 42)

  rf = RandomForestClassifier(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)

  rf.fit(X_train_lasso, y_train_lasso)

  print('Training accuracy: {:.2f} \nOOB Score: {:.2f} \nTest Accuracy: {:.2f}'.format(rf.score(X_train_lasso, y_train_lasso),
                                                                                             rf.oob_score_,
                                                                                             rf.score(X_valid_lasso, y_valid_lasso)))
