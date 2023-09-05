# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:06:31 2020

@author: yingy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:20:46 2020

@author: yingy
"""

import time	
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns 
import lightgbm as lgb
import math
from math import sqrt
from itertools import cycle
from sklearn.svm import SVR
import statsmodels.api as sm
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

%matplotlib inline 
plt.style.use('bmh')
sns.set_style("whitegrid")
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

#functions for feature engineering
def lags_windows(df):
    lags = [7]
    lag_cols = ["lag_{}".format(lag) for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id","demand"]].groupby("id")["demand"].shift(lag)

    wins = [7]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            df["rmean_{}_{}".format(lag,win)] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())   
    return df

def per_timeframe_stats(df, col):
    #For each item compute its mean and other descriptive statistics for each month and dayofweek in the dataset
    months = df['month'].unique().tolist()
    for y in months:
        df.loc[df['month'] == y, col+'_month_mean'] = df.loc[df['month'] == y].groupby(['id'])[col].transform(lambda x: x.mean()).astype("float32")
        df.loc[df['month'] == y, col+'_month_max'] = df.loc[df['month'] == y].groupby(['id'])[col].transform(lambda x: x.max()).astype("float32")
        df.loc[df['month'] == y, col+'_month_min'] = df.loc[df['month'] == y].groupby(['id'])[col].transform(lambda x: x.min()).astype("float32")
        df[col + 'month_max_to_min_diff'] = (df[col + '_month_max'] - df[col + '_month_min']).astype("float32")
    dayofweek = df['dayofweek'].unique().tolist()
    for y in dayofweek:
        df.loc[df['dayofweek'] == y, col+'_dayofweek_mean'] = df.loc[df['dayofweek'] == y].groupby(['id'])[col].transform(lambda x: x.mean()).astype("float32")
        df.loc[df['dayofweek'] == y, col+'_dayofweek_median'] = df.loc[df['dayofweek'] == y].groupby(['id'])[col].transform(lambda x: x.median()).astype("float32")
        df.loc[df['dayofweek'] == y, col+'_dayofweek_max'] = df.loc[df['dayofweek'] == y].groupby(['id'])[col].transform(lambda x: x.max()).astype("float32")
    return df

def feat_eng(df):
    df = lags_windows(df)
    df = per_timeframe_stats(df,'demand')
    return df


#imput calendar
data_calendar = pd.read_csv('C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/calendar.csv')
data_calendar.head()
calendar = data_calendar.copy()
calendar['date'] = pd.to_datetime(calendar['date'])
calendar.head()
df_2 = calendar[['date','year','month','wday']][0:1913].T

#prepare data
data_val = pd.read_csv('C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/sales_aggregation_validation.csv')
data_val.head()
sales_val = data_val.copy()

df_2.columns = sales_val.columns.values.tolist()[1:1914]
name = sales_val.columns.values.tolist()[1:1914]

#prediction settings
predictions_val = pd.DataFrame()
stats_val = pd.DataFrame(columns=['ID Name','RMSE',])

fday = datetime(2016,4,25) 
max_lags = 15
useless_cols = ['id','demand','date','demand_month_min']
linreg_train_cols = ['year','month','dayofweek','lag_7','rmean_7_7'] #use different columns for linear regression

t0 = time.time()

for i in list(range(10)):
    df_1 = sales_val[i:(i+1)]
    name_index = df_1.T.columns.values.tolist()
    id_now = str(df_1['id'].values)
    id_now = id_now.strip('[\'')
    id_now = id_now.strip('\']')
    del df_1['id']

    df_now = df_1.append(df_2)
    df_now = df_now[name]
    df_now = df_now.T
    df_now = df_now.rename(columns={i:'demand','wday':'dayofweek'})
    df_now['id'] = id_now

    #set simulated test set
    test = calendar[['date','year','month','wday']][1906:1941].copy().T
    test.columns = calendar['d'][1906:1941].T
    test = test.T
    test['demand'] = df_now['demand'][1878:1913].values
    test['id'] = id_now
    test = test.rename(columns={'wday':'dayofweek'})

    data_ml = feat_eng(df_now.copy())
    data_ml = data_ml.dropna() #!!!!

    lgb_train_cols = data_ml.columns[~data_ml.columns.isin(useless_cols)]
    x_train = data_ml[lgb_train_cols].copy()
    y_train = data_ml["demand"]

    #Fit Linear Regression
    m_rf = RandomForestRegressor(n_estimators=100,max_depth=5, random_state=26, n_jobs=-1,oob_score=True).fit(x_train, y_train)
    
    #prediction
    for tdelta in range(0, 4):
        day = fday + timedelta(days=tdelta*7+6)
        tst = test[(test.date >= day - timedelta(days=max_lags)) & (test.date <= day)].copy()
        tst = feat_eng(tst)
        tst_rf = tst.loc[(test.date >= day - timedelta(days=(max_lags-9))) & (test.date <= day) , lgb_train_cols].copy()
        tst_rf = tst_rf.fillna(0) 
        test.loc[(test.date >= day - timedelta(days=(max_lags-9))) & (test.date <= day), "preds_RandomForest"] = m_rf.predict(tst_rf)
 
    test_final = test.loc[test.date >= fday]

    #linear regression
    model_name='RandomForest'
    predictions_val[id_now] = test_final["preds_"+model_name]
    #evaluate
    score = sqrt(mean_squared_error(m_rf.oob_prediction_,x_train['lag_7'].values))
    stats_val = stats_val.append({'ID Name':id_now, 'RMSE':score},ignore_index=True)
    print(i)
    
t_1 = time.time()-t0
t_1


predictions_val = predictions_val.T
predictions_val.to_csv("/content/gdrive/My Drive/rf_validation_1.csv", index=True)
stats_val.to_csv("/content/gdrive/My Drive/RMSE_validation_1.csv", index=False)


