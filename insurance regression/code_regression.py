# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 18:36:46 2022

@author: forev
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df=pd.read_csv('insurance.csv')
# =============================================================================
# print(df.info())
# 
# for column in df.columns:
#     print(df[column].value_counts())
#     print('*'*20)
#     
# 
# =============================================================================
#plt.boxplot(df['bmi'])
#print(df['bmi'].describe())

#removing outlier using IQR

Q1=df.bmi.quantile(0.25)
Q3=df.bmi.quantile(0.75)
IQR=Q3-Q1

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR

#print(df[(df.bmi<lower_limit)|(df.bmi>upper_limit)])

df1=df[(df.bmi>lower_limit)&(df.bmi<upper_limit)]
# =============================================================================
# print(df1.info())
# for column in df1.columns:
#      print(df1[column].value_counts())
#      print('*'*20)
# 
# =============================================================================
#print(df1['charges'].describe())
df1['sex']=df1['sex'].apply({'male':0,'female':1}.get)
df1['smoker']=df1['smoker'].apply({'yes':1,'no':0}.get)
df1['region']=df1['region'].apply({'southeast':0,'southwest':1,'northeast':3,'northwest':4}.get)

#conda install spyder=5.2.2

X=df1.drop(['sex','charges'],axis=1)
y=df1['charges']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
# =============================================================================
# print(X_train.shape)
# print(X_test.shape)
# 
# =============================================================================

scaler=MinMaxScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

lr=LinearRegression()
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
print('Linear Regression R2_Score=',r2_score(y_test,pred_lr))


la=Lasso()
la.fit(X_train,y_train)
pred_la=la.predict(X_test)
print('Lasso R2_Score=',r2_score(y_test,pred_la))


ri=Ridge()
ri.fit(X_train,y_train)
pred_ri=ri.predict(X_test)
print('Ringe R2_Score=',r2_score(y_test,pred_ri))

rf=RandomForestRegressor()
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
print('Random Forest Regressor R2_Score=',r2_score(y_test,pred_rf))

fig, ax = plt.subplots(1,2, figsize=(16,6))
sns.set_style('dark')
g = sns.scatterplot(pred_lr,y_test, ax=ax[0], color='red')
g.set_title('Linear Regression')
g.set_xlabel('Predict')


sns.set_style('dark')
g = sns.scatterplot(pred_rf,y_test, ax=ax[1], color='blue')
g.set_title('Random Forest Regressor')
g.set_xlabel('Predict')

