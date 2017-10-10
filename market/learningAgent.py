from __future__ import print_function
import sys
import numpy as np
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


#Initialize first state, all items are placed deterministically
def init_state(bitcoinData, test=False):

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
    print('xdata', xdata)
    state = xdata[0:1, 0:1, :]
    print('state', state)
    sys.exit()
    return state, xdata, bitcoinData

#Take Action
def take_action(state, xdata, action, signal, time_step):
    #this should generate a list of trade signals that at evaluation time are fed to the backtester
    #the backtester should get a list of trade signals and a list of price data for the assett
    
    #make necessary adjustments to state and then return it
    time_step += 1
    
    #if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == xdata.shape[0]:
        state = xdata[time_step-1:time_step, 0:1, :]
        terminal_state = 1
        signal.loc[time_step] = 0

        return state, time_step, signal, terminal_state

    #move the market data window one step forward
    state = xdata[time_step-1:time_step, 0:1, :]
    #take action
    if action == 1:
        signal.loc[time_step] = 100
    elif action == 2:
        signal.loc[time_step] = -100
    else:
        signal.loc[time_step] = 0
    #print(state)
    terminal_state = 0
    #print('signal', signal)

    return state, time_step, signal, terminal_state

#Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)
 
    if eval == False:
        bt = twp.Backtest(pd.Series(data=[x for x in xdata[time_step-2:time_step]], index=signal[time_step-2:time_step].index.values), signal[time_step-2:time_step], signalType='shares')
        reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])

    if terminal_state == 1 and eval == True:
        #save a figure of the test set
        bt = twp.Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]
        plt.figure(figsize=(3,4))
        bt.plotTrades()
        plt.axvline(x=400, color='black', linestyle='--')
        plt.text(250, 400, 'training data')
        plt.text(450, 400, 'test data')
        plt.suptitle(str(epoch))
        plt.savefig('plt/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
        plt.close('all')
    #print(time_step, terminal_state, eval, reward)

    return reward



def buyBitcoin(eval_model, globalPriceHistory, epoch=0):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    eval_data = globalPriceHistory
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = init_state(eval_data)
  
    status = 1
    terminal_state = 0
    time_step = 1
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True, epoch=epoch)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0

    return eval_reward

