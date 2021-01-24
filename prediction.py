# dependencies
import numpy as np;

import pandas as pd;
from pandas.plotting import lag_plot;
from pandas import datetime;

import math;
import matplotlib.pyplot as plt;
from statsmodels.tsa.arima.model import ARIMA;
from sklearn.metrics import mean_squared_error;

# redundant imports
#import os;
#from subprocess import check_output;
#import seaborn as sns;
#import warnings;

# constants
TEST = 'TEST'
PREDICT = 'PREDICT'

# file to read
df = pd.read_csv('./HDFCBANK.NS.csv')

# config switches
tick_int = round(len(df) / 10)
lag = 5
days_to_predict = 50
mode = TEST # TEST / PREDICT
interim_plots = False
split_testing_data_at = 0.9
tup = (2,3,2)

# helper functions
def smape_kun(y_true, y_pred):
   return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))
# function to return index array of x axis for predicted data
def temp_func(index_array, base_data):
	returnVar = list()
	for i in range(len(index_array)):
		returnVar.append(i+len(base_data))
	return returnVar

# plotting of existing data
if interim_plots == True:
	print('Plotting existing data on chart ...')
	plt.figure(figsize=(7,7))
	plt.plot(df[['Close']], 'green', label='Base Ticker Data')
	plt.title("HDFC Bank")
	plt.draw()

# plotting of autocorrelation chart
if interim_plots == True:
	print('Plotting autocorrelation ...')
	plt.figure(figsize=(7,7))
	lag_plot(df['Open'], lag)
	plt.title('HDFC Autocorrelation plot')
	plt.draw()

# switches for train data / test data population (initialise data)
print('Initialising data ...')
if mode == PREDICT:
	train_data = df
	#train_data = df[0:int(len(df)*split_testing_data_at)]
if mode == TEST:
	train_data, test_data = df[0:int(len(df)*split_testing_data_at)], df[int(len(df)*split_testing_data_at):]

# plotting of training / testing data
if interim_plots == True:
	print('Plotting training / testing data ...')
	plt.figure(figsize=(7,7))
	plt.title('HDFC Bank Prices')
	plt.xlabel('Dates')
	plt.ylabel('Prices')
	plt.plot(df['Open'], 'blue', label='Training Data')
	plt.plot(test_data['Open'], 'green', label='Testing Data')
	plt.xticks(np.arange(0,len(df),tick_int ), df['Date'][0:len(df):tick_int])
	plt.legend()
	plt.draw()

# common for training and prediction
predictions = list()
train_ar = train_data['Open'].values
if mode == TEST:
	test_ar = test_data['Open'].values
history = [x for x in train_ar]

print('Running analysis ...')

# training and validation
if mode == TEST:
	model = ARIMA(history, order=tup)
	model_fit = model.fit()
	output = model_fit.forecast(len(test_ar))

	# for t in range(len(test_ar)):
	#     model = ARIMA(history, order=(10,1,3))
	#     model_fit = model.fit()
	#     output = model_fit.forecast()
	#     yhat = output[0]
	#     predictions.append(yhat)
	#     print(t, yhat)
	#     obs = yhat
	#     #obs = test_ar[t]
	#     history.append(obs)

# prediction
if mode == PREDICT:
	model = ARIMA(history, order=tup)
	model_fit = model.fit()
	output = model_fit.forecast(days_to_predict)

	# for t in range(days_to_predict):
	# 	model = ARIMA(history, order=tup)
	# 	model_fit = model.fit()
	# 	output = model_fit.forecast()
	# 	yhat = output[0]
	# 	print(t, yhat)
	# 	predictions.append(yhat)
	# 	obs = yhat
	# 	history.append(obs)

#common end functions
for t in output:
	yhat = t
	predictions.append(yhat)
	print(t, yhat)
	obs = yhat
	history.append(obs)

# test results
if mode == TEST:
	error = math.sqrt(mean_squared_error(test_ar, predictions))
	print('Testing Mean Squared Error: %.3f' % error)
	error2 = smape_kun(test_ar, predictions)
	print('Symmetric mean absolute percentage error: %.3f' % error2)

# plot
print('Plotting results ...')
plt.figure(figsize=(7,7))

if mode == TEST:
	plt.plot(df['Open'], 'green', color='blue', label='Training Data')
	plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
	plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')

if mode == PREDICT:
	plt.plot(train_data['Open'], 'green', color='blue', label='Historical Data')
	plt.plot(temp_func(predictions, train_ar), predictions, color='red', label='Predicted Price')

plt.title('HDFC Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0,len(df)-1, tick_int), df['Date'][0:len(df)-1:tick_int])
plt.legend()
plt.show()
