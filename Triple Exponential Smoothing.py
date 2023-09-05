# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:27:43 2020

@author: yingy
"""

#import packages and plot setting
import time	
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns 
import math
from math import sqrt
from itertools import cycle
from sklearn.svm import SVR
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
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


#loss for the median
def spl_denominator(train_series):
    N = len(train_series)
    sumup = 0
    for n in range(1, N):
        sumup += np.abs(train_series[n]-train_series[n-1])
    return sumup/(N-1)

def spl_numerator(dev_series, Q, u):
    sumup = 0
    for m in range(len(dev_series)):
        if Q[m] <= dev_series[m]:
            sumup += (dev_series[m] - Q[m])*u
        else:
            sumup += (Q[m] - dev_series[m])*(1-u)
    return sumup

def spl(train_series, dev_series, Q, u):
    h = len(dev_series)
    spl_denomina = spl_denominator(train_series)
    spl_numera = spl_numerator(dev_series, Q, u)
    
    return spl_numera/(h*spl_denomina)


#----------Calculation: validation----------

#input data
data2 = pd.read_csv('C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/sales_aggregation_validation.csv')
data2.head()

data_id = data2['id']
data = data2.copy()
del data['id']
data.head()

#prediction settings
predictions = pd.DataFrame()
stats = pd.DataFrame(columns=['ID Name','RMSE'])

#prepare data
row_now = data[0:1].T
name = row_now.columns.values.tolist()

#'Triple Exponential Smoothing'
id_name = str(data_id[0:1].values)
id_name_new = id_name.strip('[\'')
id_name_new = id_name_new.strip('\']')
#train
tripleExpSmooth_model = ExponentialSmoothing(row_now[name].values,trend='add',seasonal='add',seasonal_periods=7).fit()
#predict
predictions[id_name_new] = tripleExpSmooth_model.forecast(28)#.values
#evaluate
score = sqrt(tripleExpSmooth_model.sse/1913)
stats = stats.append({'ID Name':id_name_new, 'RMSE':score},ignore_index=True)


for i in list(range(42840)):
    row_now = data[i:(i+1)].T
    name = row_now.columns.values.tolist()
    id_name = str(data_id[i:(i+1)].values)
    id_name_new = id_name.strip('[\'')
    id_name_new = id_name_new.strip('\']')
    tripleExpSmooth_model = ExponentialSmoothing(row_now[name].values,trend='add',seasonal='add',seasonal_periods=7).fit()
    predictions[id_name_new] = tripleExpSmooth_model.forecast(28)#.values
    score = sqrt(tripleExpSmooth_model.sse/1913)
    stats = stats.append({'ID Name':id_name_new, 'RMSE':score},ignore_index=True)
    print(i)

predictions_new = predictions.copy()
predictions_new = predictions_new.T
predictions_new.to_csv("C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/some results/Triple Exponential Smoothing/point_prediction_validation.csv", index=True)
stats_new = stats.copy()
stats_new.to_csv("C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/some results/Triple Exponential Smoothing/RMSE_validation.csv", index=False)



#----------Calculation: evaluation----------

#input data
data1 = pd.read_csv('C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/sales_aggregation_evaluation.csv')
data1.head()

data_id = data1['id']
data = data1.copy()
del data['id']
data.head()

#prediction settings
predictions_e = pd.DataFrame()
stats_e = pd.DataFrame(columns=['ID Name','RMSE'])

for i in list(range(42840)):
    row_now = data[i:(i+1)].T
    name = row_now.columns.values.tolist()
    id_name = str(data_id[i:(i+1)].values)
    id_name_new = id_name.strip('[\'')
    id_name_new = id_name_new.strip('\']')
    tripleExpSmooth_model = ExponentialSmoothing(row_now[name].values,trend='add',seasonal='add',seasonal_periods=7).fit()
    predictions_e[id_name_new] = tripleExpSmooth_model.forecast(28)#.values
    score = sqrt(tripleExpSmooth_model.sse/1941)
    stats_e = stats_e.append({'ID Name':id_name_new, 'RMSE':score},ignore_index=True)
    print(i)

predictions_new = predictions_e.copy()
predictions_new = predictions_new.T
predictions_new.to_csv("C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/some results/Triple Exponential Smoothing/point_prediction_evaluation.csv", index=True)
stats_new = stats_e.copy()
stats_new.to_csv("C:/Users/yingy/Desktop/research/Kaggle - M5 Forecasting/data/some results/Triple Exponential Smoothing/RMSE_evaluation.csv", index=False)



