# library for dataset manipulation
import pandas as pd
import numpy as np
from numpy import array
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis, skew
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier


# import csv and saved into dataframe
df = pd.read_csv("tblJDataGuardian.csv")
# print first 5 row

print("=== DF ===")
print(df)
print("==========")

# change 0.00 data to nan
df['BGLevel'].replace(0.00, np.nan, inplace=True)
# interpolated nan data
df['BGLevel'].interpolate(method='bfill',inplace=True)
# Manipulated BG date string to delete the time using map lambda Function
df['BGDate'] = df['BGDate'].map(lambda x: x.replace(' 00:00:00', ''))
df['BGDateTime'] = df['BGDate'] + ' ' + df['BGTime']

def convertDateTimeStringtoDateTimeObject(strDateTime):
    date_time_obj = datetime.strptime(strDateTime, '%Y-%m-%d %I:%M:%S %p')
    return date_time_obj
df['BGDateTime'] = df['BGDateTime'].map(convertDateTimeStringtoDateTimeObject)

new_df = df[['RecID','PtID','BGDateTime','BGLevel']].copy()


print("=== new_df ===")
print(new_df)
print("==========")

def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2=list()
        inp2=input_data[i]
        min=float(np.min(inp2))
        max=float(np.max(inp2))
        diff=(max-min)
        std=float(np.std(inp2))
        mean=float(np.mean(inp2))
        median=float(np.median(inp2))
        kurt=float(kurtosis(inp2))
        sk=float(skew(inp2))
        inp2=np.append(inp2,min)
        inp2=np.append(inp2,max)
        inp2=np.append(inp2,diff)
        inp2=np.append(inp2,std)
        inp2=np.append(inp2,mean)
        inp2=np.append(inp2,median)
        inp2=np.append(inp2,kurt)
        inp2=np.append(inp2,sk)
        #print(list(inp2))
        inp=np.append(inp,inp2)    
    # print("stats_features")
    # print(inp)
    inp=inp.reshape(len(input_data),-1)
    #print(inp)
    return inp

def split_sequence(sequence, n_steps, ph):
    X, y = list(), list()
    for i in range(len(sequence)):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the sequence
        if end_ix > len(sequence) - ph:
            break
		# gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix + ph - 1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def classifiers(X_train, y_train_ori, X_test, y_test_ori):
    print("start classifiers")
    print("fit and prediction models")
    # Linear Regression
    lr = linear_model.LinearRegression()
    lr.fit(X_train,  y_train_ori)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = sqrt(mean_squared_error(y_test_ori, y_pred_lr))

    # Random Forest
    rf = RandomForestRegressor()
    rf.fit(X_train,  y_train_ori)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = sqrt(mean_squared_error(y_test_ori, y_pred_rf))

    # XGB
    mxgb = xgb.XGBRegressor()
    mxgb.fit(X_train,  y_train_ori)
    y_pred_xgb = mxgb.predict(X_test)
    rmse_xgb = sqrt(mean_squared_error(y_test_ori, y_pred_xgb))

    # Decission Tree
    tree = DecisionTreeRegressor(max_depth = 2)
    tree.fit(X_train,  y_train_ori)
    y_pred_tree = tree.predict(X_test)
    rmse_tree = sqrt(mean_squared_error(y_test_ori, y_pred_tree))

    #SVR
    svr = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
    svr.fit(X_train,  y_train_ori)
    y_pred_svr = svr.predict(X_test)
    rmse_svr = sqrt(mean_squared_error(y_test_ori,y_pred_svr))
    
    # Lasso Regression
    lasso = linear_model.Lasso(alpha=1.0)
    lasso.fit(X_train,  y_train_ori)
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = sqrt(mean_squared_error(y_test_ori, y_pred_lasso))
    
    # Ridge Regression
    ridge = linear_model.Ridge(alpha=1.0)
    ridge.fit(X_train,  y_train_ori)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = sqrt(mean_squared_error(y_test_ori, y_pred_ridge))

    # KNN
    # knn = KNeighborsClassifier()
    # knn.fit(X_train,  y_train_ori)
    # y_pred_knn = knn.predict(X_test)
    # rmse_knn = sqrt(mean_squared_error(y_test_ori, y_pred_knn))

    print("end classifiers")
    return rmse_lr, rmse_rf, rmse_xgb, rmse_tree, rmse_svr, rmse_lasso, rmse_ridge
  

print("Train Test Split")
index_x = 0

arr_patient_id = [1,15,4,21]
train_size = 0.8
window = 6
ph = 6

patient_data_train_test = list()
patient_data_train_test_with_stats = list()

for i in arr_patient_id:
  patient_id=i
  df_patient = new_df[new_df['PtID']==patient_id]
  # df_patient.plot(x = 'BGDateTime', y = 'BGLevel', title='Patient ID : '+str(patient_id))
  data_patient = df_patient['BGLevel'].astype(float).values
  len_data_patient = len(data_patient)
  len_train = int(math.ceil(len_data_patient * train_size))

  data_train = data_patient[0:len_train]
  data_test = data_patient[len_train:]

  X_train,y_train_ori = split_sequence(data_train, window, ph)
  X_test,y_test_ori = split_sequence(data_test, window, ph)
  
  print("Train Test Split of patient_id : ",patient_id)
  X_train_stats = stats_features(X_train)
  X_test_stats = stats_features(X_test)

  patient_data_train_test = np.append(patient_data_train_test,{
      "patient_id": patient_id,
      "X_train": X_train,
      "y_train_ori": y_train_ori,
      "X_test": X_test,
      "y_test_ori": y_test_ori,
  })
  patient_data_train_test_with_stats = np.append(patient_data_train_test_with_stats,{
      "patient_id": patient_id,
      "X_train": X_train_stats,
      "y_train_ori": y_train_ori,
      "X_test": X_test_stats,
      "y_test_ori": y_test_ori,
  })


print("End Train Test Split ==============================")
  

# print(patient_data_train_test)

for patient in patient_data_train_test:
  print("Patient ID : " + str(patient['patient_id']))
  rmse_lr, rmse_rf, rmse_xgb, rmse_tree, rmse_svr, rmse_lasso, rmse_ridge = classifiers(patient['X_train'],patient['y_train_ori'],patient['X_test'],patient['y_test_ori'])
  print("RMSE Linear Regression : ")
  print(rmse_lr)
  print("RMSE Random Forest : ")
  print(rmse_rf)
  print("XGBoost : ")
  print(rmse_xgb)
  print("Decission Tree : ")
  print(rmse_tree)
  print("SVR : ")
  print(rmse_svr)
  print("Lasso Regression: ")
  print(rmse_lasso)
  print("Ridge Regression : ")
  print(rmse_ridge)
  print("=========================")
  # axis[0, index_x].plot(X_test, y_test_ori)
  # axis[0, index_x].plot(X_test, y_pred_lr)
  # axis[0, index_x].set_title("Patient ID : " + str(patient['patient_id']))

  index_x += 1

