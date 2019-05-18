# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:52:16 2019

@author: Computer
"""

import pandas as pd
import regex
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Data dict url:
#https://chronicdata.cdc.gov/500-Cities/500-Cities-City-level-Data-GIS-Friendly-Format-201/dxpw-cm5u
df = pd.read_csv(r'C:\Users\Computer\Desktop\School\csv\500_Cities_Data.csv')
#maybe use state as predictor
state_to_dummy = df['StateAbbr']
#%%
df = df.drop(columns = ['PlaceName', 'Geolocation', 'PlaceFIPS', 'StateAbbr'], axis =1)
#%%

#Purpose: create algorithm to determine healthcare access.

#only want crude estimates that takes percent of population
df = df[df.columns.drop(list(df.filter(regex='95')))]
#%%
df = df[df.columns.drop(list(df.filter(regex='AdjPrev')))]

for cols in df.columns:
    print(cols)
#%%
print(df.ACCESS2_CrudePrev.describe())
y_var = df.ACCESS2_CrudePrev
plt.hist(y_var)
#left skewed, we can normalize once we explore more
#%%
corr = df.corr()
#we have a lot of variables that correlate here, which makes sense
#Let's drop ally_var columns that have .6 corr or higher w. 2 or more columns
df = df.drop(columns = ['CHD_CrudePrev', 'ARTHRITIS_CrudePrev', 'BPHIGH_CrudePrev', 'COPD_CrudePrev',
                        'LPA_CrudePrev', 'KIDNEY_CrudePrev'])
#%%
#let's create a flat feature for all crude prev of check ups and then drop
df['CORE_AVG'] = (df['COREM_CrudePrev'] + df['COREW_CrudePrev']) / 2
df = df.drop(columns = ['COREM_CrudePrev', 'COREW_CrudePrev', 'COLON_SCREEN_CrudePrev'])
#%%
df = df.drop(columns = ['TEETHLOST_CrudePrev', 'CSMOKING_CrudePrev',
             'OBESITY_CrudePrev','PHLTH_CrudePrev','MHLTH_CrudePrev',
             'STROKE_CrudePrev', 'DIABETES_CrudePrev', 'DENTAL_CrudePrev',
             'BPMED_CrudePrev', 'PAPTEST_CrudePrev'])
#%%
df['Population2010'] = np.log(df['Population2010'])
#%%
#plot distributions for variables

plt.figure(figsize=(9,15))
i = 1
for column in df.columns:
    plt.subplot(5,4,i)
    sns.distplot(df[column])
    i+=1
plt.tight_layout()
plt.show()
#looks mostly normal
#%%
#test new correrlation table
corr = df.corr()
#%%
#create target
y_var = np.array(y_var)
y_var[y_var <= 15 ] = 0
y_var[y_var > 15] = 1
df = df.drop(columns =['ACCESS2_CrudePrev'])
#%%
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df
lr = LogisticRegression(C=1e9)

X_train, X_test, y_train, y_test = train_test_split(X,y_var, test_size = .3)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print('Logistic regression training accuracy score: ', round(lr.score(X_train, y_train), 3))
print('Logistic regression test accuracy score: ', round(lr.score(X_test, y_test), 3))
#looks decent
#use cross validation
#try to winsorize
#look for heteroscedasticity, box cox
#%%
describe = df.describe()
#to_excel