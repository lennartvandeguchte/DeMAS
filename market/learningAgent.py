from __future__ import print_function
import sys
import numpy as np
from numpy import newaxis
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
import talib as tl
from sklearn.externals import joblib

import quandl

def buyBitcoin(eval_model, globalPriceHistory, time):
    #signal = pd.Series(index=np.arange(415))
    state, xdata, price_data = init_state(globalPriceHistory)


    qval = eval_model.predict(state, batch_size=1)
    print('qval', qval)
    action = (np.argmax(qval))
    print('action', action)

    return action

#Initialize first state, all items are placed deterministically
def init_state(bitcoinData, test=True):

    bitcoinData = np.array(bitcoinData)
    diff = np.diff(bitcoinData) #compute differences bitcoinprices between days
    diff = np.insert(diff, 0, 0)

    sma15 = tl.SMA(bitcoinData, timeperiod=15)   #computes simple moving average over timeperiods of 15/60 days  
    sma60 = tl.SMA(bitcoinData, timeperiod=60)
    rsi = tl.RSI(bitcoinData, timeperiod=14)   #Computes the relative strength index (rsi)
    #atr = tl.ATR(close, timeperiod=14)   #Computes the average true rate (atr)

    #--- Preprocess data
    xdata = np.column_stack((bitcoinData, diff, sma15, bitcoinData-sma15, sma15-sma60, rsi))

    xdata = np.nan_to_num(xdata)

    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    elif test == True:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)

    state = xdata[-1:, -1:, :]
    print('state', state)

    return state, xdata, bitcoinData


    


