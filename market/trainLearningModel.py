''' This code is made with help of the tutorial of Daniel Zakrisson '''

from __future__ import print_function
from functionsLearningAgent import *
import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.externals import joblib
import sys



###### Neural network ####################################################################
# The neural learns to represent the states in which te agent can be
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

tsteps = 1
batch_size = 10
num_features = 6

model = Sequential()
model.add(LSTM(64,
               input_shape=(1, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.5))

model.add(LSTM(64,
               input_shape=(1, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.5))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear')) 

rms = RMSprop()
adam = Adam()
model.compile(loss='mse', optimizer=adam)



##### Script to run ########################################################################
import random, timeit
start_time = timeit.default_timer()

simulated_data = load_simulated_data()
test_data = simulated_data['results%s' %(len(simulated_data)-1)]

# Do Q-learning in combination with a neural network on simulted data
for index in range(len(simulated_data)-1):
    #Load one instance of the simulated data
    bitcoinData = simulated_data['results%s' %index]

    # Settings
    epochs = 4
    gamma = 0.8 # A high gamma will let the Q-learning search for rewards more time steps away
    epsilon = 1
    batchSize = 50
    buffer = 100
    replay = []
    learning_progress = [] # Stores tuples of (S, A, R, S')
    h = 0
    signal = pd.Series(index=np.arange(len(bitcoinData)))
  

    for i in range(epochs):
        state, xdata, price_data = init_state(bitcoinData)

        status = 1
        terminal_state = 0
        time_step = 1

        while(status == 1):
            #We are in state S, do the Q function on S to get Q values for all possible actions
            qval = model.predict(state, batch_size=1)
    
            if (random.random() < epsilon): # Choose random action
                action = np.random.randint(0,4) # Assumes 4 different actions (buy, sell and two times pass)
            else: # Choose best action from Q(s,a) values
                action = (np.argmax(qval))
            # Take action, observe new state S'
            new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
            # Observe reward
            reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state)

            # Experience replay storage
            if (len(replay) < buffer): 
                replay.append((state, action, reward, new_state))
            else: # If buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    old_qval = model.predict(old_state, batch_size=1)
                    newQ = model.predict(new_state, batch_size=1)
                    maxQ = np.max(newQ)
                    y = np.zeros((1,4))
                    y[:] = old_qval[:]
                    if terminal_state == 0: # Non-terminal state
                        update = (reward + (gamma * maxQ))
                    else: # Terminal state
                        update = reward
                    y[0][action] = update
                    X_train.append(old_state)
                    y_train.append(y.reshape(4,))

                X_train = np.squeeze(np.array(X_train), axis=(1))
                y_train = np.array(y_train)
                model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=0)
                
                state = new_state
            if terminal_state == 1:
                status = 0
        
        if epsilon > 0.1: # Decrement epsilon over time
            epsilon -= (1.0/epochs)

    eval_reward = evaluate_Q(test_data, model, price_data, i)
    learning_progress.append((eval_reward))


# Save the model to json file
model_json = model.to_json()
with open("inputs\model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("inputs\model.h5")
print("Saved model to json file")

elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

bt = twp.Backtest(pd.Series(data=[x[0,0] for x in xdata]), signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)
print('bt.data', bt.data)

print(bt.data)
unique, counts = np.unique(filter(lambda v: v==v, signal.values), return_counts=True)
print('signal values', signal.values)
print(np.asarray((unique, counts)).T)

# Plot results
plt.figure()
plt.subplot(2,1,1)
bt.plotTrades()
plt.subplot(2,1,2)
bt.pnl.plot(style='x-')

plt.savefig('plt/summary'+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.show()


