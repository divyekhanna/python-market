# dependencies
import numpy as np;

import pandas as pd;
from pandas.plotting import lag_plot;
from pandas import datetime;

import math;
import matplotlib.pyplot as plt;
from statsmodels.tsa.arima.model import ARIMA;
from sklearn.metrics import mean_squared_error;

#import os;
#from subprocess import check_output;
#import seaborn as sns;
#import warnings;

# file to read
df = pd.read_csv('./HDFCBANK.NS.csv')

# config
tick_int = round(len(df) / 10)
lag = 2

# plotting of existing data
#df[['Close']].plot()
#plt.title("HDFC Bank")
#plt.show()

# plotting of y(t) against y(t+5)
#plt.figure(figsize=(10,10))
#lag_plot(df['Open'], lag)
#plt.title('HDFC Autocorrelation plot')
#plt.show()

# model training for testing
#df['Date'][len(df)-1]
#train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
#plt.figure(figsize=(7,7))
#plt.title('HDFC Bank Prices')
#plt.xlabel('Dates')
#plt.ylabel('Prices')
#plt.plot(df['Open'], 'blue', label='Training Data')
#plt.plot(test_data['Open'], 'green', label='Testing Data')
#plt.xticks(np.arange(0,len(df),tick_int ), df['Date'][0:len(df):tick_int])
#plt.legend()

#def smape_kun(y_true, y_pred):
#    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))
train_data = df
train_ar = train_data['Open'].values
# test_ar = test_data['Open'].values
#print (train_ar)
# print (test_ar)
history = [x for x in train_ar]
#print(type(history))
# print (history)
predictions = list()
# for t in range(len(test_ar)):
#     model = ARIMA(history, order=(5,1,0))
#     model_fit = model.fit()
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test_ar[t]
#     history.append(obs)
#     print (t)

for t in range(200):
	print (t)
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	print(yhat)
	predictions.append(yhat)
	obs = yhat
	history.append(obs)
	
# error = math.sqrt(mean_squared_error(test_ar, predictions))
# print('Testing Mean Squared Error: %.3f' % error)
# error2 = smape_kun(test_ar, predictions)
# print('Symmetric mean absolute percentage error: %.3f' % error2)
#print (history)

# function to return index array of x axis for predicted data
def temp_func(index_array):
	returnVar = list()
	for i in range(len(index_array)):
		returnVar.append(i+len(df))
	return returnVar

plt.figure(figsize=(7,7))
plt.plot(df['Open'], 'green', color='blue', label='Historical Data') #Training Data
plt.plot(temp_func(predictions), predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')

# plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.title('HDFC Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0,len(df)-1, tick_int), df['Date'][0:len(df)-1:tick_int])
plt.legend()
plt.show()
