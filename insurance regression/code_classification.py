# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:35:02 2022

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



X=df1.drop('smoker',axis=1)
y=df1['smoker']

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

scaler=StandardScaler()
scaler.fit(X)
scaled_features=scaler.transform(X)
X=pd.DataFrame(scaled_features)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

svc=SVC()
svc.fit(X_train,y_train)
y_pred_svc=svc.predict(X_test)
print('Support Vector Machines Classifier = ',accuracy_score(y_test, y_pred_svc))

neigh=KNeighborsClassifier()
neigh.fit(X_train,y_train)
y_pred_neigh=neigh.predict(X_test)
print('K Neighbor Classifier = ',accuracy_score(y_test, y_pred_neigh))

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)
print('Decision Tree Classifier = ',accuracy_score(y_test, y_pred_dt))

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print('Random Forest Classifier = ',accuracy_score(y_test, y_pred_rf))


lda=LDA()
lda.fit(X_train,y_train)
y_pred_lda=lda.predict(X_test)
print('Linear Discriminant Analysis = ',accuracy_score(y_test, y_pred_lda))

print('*'*20)
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test,y_pred_svc))

