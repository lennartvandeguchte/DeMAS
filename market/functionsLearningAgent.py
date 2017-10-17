from __future__ import print_function

import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.externals import joblib
import sys
import quandl

#Load real bitcoin data
def read_convert_data(symbol='BITSTAMP'):
    if symbol == 'BITSTAMP':
        prices = quandl.get("BCHARTS/BITSTAMPUSD")
        prices.to_pickle('data/BITSTAMP_1day.pkl') # a /data folder must exist
    #print(prices)

def load_data(test=False):
    prices = pd.read_pickle('data/BITSTAMP_1day.pkl')
    prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume (BTC)': 'volume'}, inplace=True)
    x_train = prices.iloc[-2102:-1271,]
    x_test= prices.iloc[-1321:-1221,]
    print(x_train)
    if test:
        return x_test
    else:
        return x_train

# Load simulated bitcoin data
def load_simulated_data():
    return pd.read_pickle('data/simulationPricesTEST5.pkl')
   

#Initialize first state, all items are placed deterministically
def init_state(bitcoinData, test=False):
    
    # Compute features of bitcoin price to feed the network
    diff = np.diff(bitcoinData) #Difference of bitcoinprices between days
    diff = np.insert(diff, 0, 0)
    cumsum_vec = np.cumsum(np.insert(bitcoinData, 0, 0)) 
    sma15 = (cumsum_vec[15:] - cumsum_vec[:-15]) / 15 #computes simple moving average over timeperiods of 15/60 days  
    
    if(test==False or (len(bitcoinData)>14)):
        for i in range(14):
            sma15 = np.insert(sma15, 0, np.nan)

    sma60 = (cumsum_vec[60:] - cumsum_vec[:-60]) / 60
    if(test==False or (len(bitcoinData)>59)):
        for i in range(59):
            sma60 = np.insert(sma60, 0, np.nan)
        
    rsi = rsiFunc(bitcoinData, diff)   #Computes the relative strength index (rsi)

    #--- Preprocess data
    if(test==True):
        if(len(bitcoinData) < 15):
            for i in range(len(bitcoinData)):
                sma15 = np.insert(sma15, 0, np.nan)
        
        if(len(bitcoinData) < 60):
            for i in range(len(bitcoinData)):
                sma60 = np.insert(sma60, 0, np.nan)   

    xdata = np.column_stack((bitcoinData, diff, sma15, bitcoinData-sma15, sma15-sma60, rsi))
    xdata = np.nan_to_num(xdata)

    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    elif test == True:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    
    state = xdata[0:1, 0:1, :]
    return state, xdata, bitcoinData

# Function to compute the relative strength index
def rsiFunc(prices, diff, n=14):
    seed = diff[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = diff[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

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

def evaluate_Q(eval_data, eval_model, price_data, epoch=0):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
  
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = init_state(eval_data)
    status = 1
    terminal_state = 0
    time_step = 1
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        #print('qval', qval)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True, epoch=epoch)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0

    return eval_reward

def buySellOrPass(eval_data, eval_model): 
    state, xdata, price_data = init_state(eval_data, test=True)
    print('state', state)
    qval = eval_model.predict(state, batch_size=1)
    action = (np.argmax(qval))
    return action
