# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:35:32 2020

@author: Abhishek
"""
import datetime
import os
import xlrd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVR
import pandas as pd
pd.set_option('display.max_columns', None)
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#%%

wb=xlrd.open_workbook('newdata.xlsx', logfile=open(os.devnull, 'w'))
data1=pd.read_excel(wb)
data1.dropna(how='all', axis=1,inplace=True)
X=data1.drop(["CAST_ID","TIME_STAMP","TEMP_DIFF","DUR_T8"],axis=1)
X=X.fillna(0)
X.reset_index(inplace = True,drop = True)

y = data1.TEMP_DIFF
y=y.fillna(0)
y.reset_index(inplace = True,drop = True)

new_dur_t=[]
new_temp=data1.LAST_TEMP

for i in range(0,len(X)-1):
    temp1=new_temp[i+1]
    temp2=new_temp[i]
    temp_diff=temp1-temp2 
    new_dur_t.insert(i,temp_diff)
    
new_dur_t.insert(len(X)-1,0)
X['NEW_TEMP']=new_dur_t
X.NEW_TEMP=X.NEW_TEMP.fillna(0)
'''
print("Which scaler do you want to use?\n")
scale_ch=int(input("1. Standard Scaler\n2. MinMaxScaler\n3. RobustScaler\n4. Normalizer\n5. No Selection(use as it is)\n"))

if(scale_ch==1):
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    
elif(scale_ch==2):
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)

elif(scale_ch==3):
    scaler=RobustScaler()
    X=scaler.fit_transform(X)
    
elif(scale_ch==4):
    scaler=Normalizer()
    X=scaler.fit_transform(X)

X=pd.DataFrame(X)
y=pd.DataFrame(y)
'''

df = pd.concat([X, y], axis=1)
print(df.describe())

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

#%%

pp = PdfPages('newDataFeatures.pdf')
for i in X.columns:
    if(i=="NEW_TEMP"):
     break
    plot1=plt.figure()
    plt.xlabel(i,fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plot2=plt.hist(data1[i])
    pp.savefig(plot1)
    plt.show()
pp.close()

#%%

print("Which regression technique do you want do apply?")
regr_ch=int(input("1. Linear Regresion\n2. Linear Regression (Lasso)\n3. Linear Regression (Ridge)\n4. Support Vector Regression\n5. Random Forest Regressor\n6. XGBoost Regressor(recommended)\n"))

print("Which technique do you want to use for feature selection?\n")
feat_ch=int(input("1. SelectKBest\n2. Correlation Matrix(recommended)\n3. No Selection(use all features)\n"))

#%%
if(feat_ch==2):
 e=float(input("Select % threshold (Eg. 0.1 means only the features that have atleast 10% impact will be selected)\n"))
 plt.figure(figsize=(20,20))
 cor = df.corr()
 sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
 plt.show()
 cor_target = abs(cor["TEMP_DIFF"])
 relevant_features = cor_target[cor_target>e]
 X=df[relevant_features.index]
 X=X.drop(["TEMP_DIFF"],axis=1)
 X=X.fillna(0)
 print("The DataSet X after selecting relevant features is:")
 print(X.describe())

if(feat_ch==1):
 c=int(input("Enter the value of K for SelectKBest\n"))
 d=int(input("Select scoring method:\n1. Mutual info regression\n2. F-value\n"))
 if(d==1):
  X=SelectKBest(score_func=mutual_info_regression,k=c).fit_transform(X,y)
 else:
  X=SelectKBest(score_func=f_regression,k=c).fit_transform(X,y)

#%%

if(regr_ch==6):
 regr=xgb.XGBRegressor()

#%%

if(regr_ch==5):
 o=int(input("Enter the number of estimators\n"))
 regr=RandomForestRegressor(n_estimators = o, random_state = 47)

#%%

if regr_ch==2:
 alpha1=float(input("Enter the value of alpha\n"))
 regr=linear_model.Lasso(alpha=alpha1)
if regr_ch==3:
 alpha1=float(input("Enter the value of alpha\n"))
 regr=linear_model.Ridge(alpha=alpha1)
if regr_ch==1:
 regr=LinearRegression()

#%%

if(regr_ch==4):
 regr=svm.SVR()

a=float(input("Enter % of test size for splitting (in decimal between 0 to 1 (Eg. 0.2 for 20%))\n"))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=a,random_state=48)

#%%
 
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)

plt.scatter(y_pred, y_test) 
plt.show()


plt.bar(y_pred, y_test) 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.show()

#%%
if(regr_ch==6):
 xgb.plot_importance(regr)
 plt.rcParams['figure.figsize'] = [20, 20]
 plt.show()
 print(regr.feature_importances_)
 feat_importances=pd.Series(regr.feature_importances_,index=X.columns)
 q=int(input("How many features you want to plot?"))
 feat_importances.nlargest(q).plot(kind='barh')
 plt.show()

elif(regr_ch==5):
 print(regr.feature_importances_)
 feat_importances=pd.Series(regr.feature_importances_,index=X.columns)
 q=int(input("How many features you want to plot?"))
 feat_importances.nlargest(q).plot(kind='barh')
 plt.show()

#%%

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(regr.score(X_test,y_test))

#%%

score_mse = cross_val_score(regr, X, y, cv=5, scoring='neg_mean_squared_error')
score_mae = cross_val_score(regr, X, y, cv=5, scoring='neg_mean_absolute_error')
score_rmse = cross_val_score(regr, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("After applying 5-fold cross_validation:")
print("MAE:")
print(-1*score_mae.mean())
print("MSE:")
print(-1*score_mse.mean())
print("RMSE:")
print(-1*score_rmse.mean())

#%%