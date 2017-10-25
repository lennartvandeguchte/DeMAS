from model import *
import matplotlib.pyplot as plt
import numpy
import json
import pickle

# Settings
amountSteps = 415
amountRuns = 1
all_history = numpy.zeros(415)
all_history_save = {}
save_simulations = False
train_learning_agent = False
include_learning_agent = True
include_fast_trader = True

# Amount of simulations
for j in range(amountRuns):
    # Create new model for a simulation
    model = Market(include_learning_agent, include_fast_trader)

    # Max range (timesteps) equals 415 (830 days), this simulates the period between 01-01-2012 and 10-04-2014
    for i in range(amountSteps):
        model.step()
        all_history[i] += model.globalPriceHistory[i]
    # Show results for simulation j after 415 timesteps
    all_history_save['results%s' % j]= model.globalPriceHistory


# Save simulated data to train and test the learning agent
if(save_simulations):   
    with open('data/simulationPricesTEST5.pkl', 'wb') as f:
        pickle.dump(all_history_save, f, pickle.HIGHEST_PROTOCOL)

# Set train_learning_agent on True to let the agent learn, afterwards it can be included in the simulation
# This file can also be runned seperately to tune parameters
if(train_learning_agent):   
    exec(open("trainLearningModel.py").read()) 


# Show the average simulated bitcoin price over an amount of runs
for h in range(amountSteps):
    all_history[h] = all_history[h]/amountRuns
fig1 = plt.figure()   
plt.plot(all_history)


# Plot the average virtual wealth of each type of agent
fig2 = plt.figure()
ax1 = fig2.add_subplot(111) 
plt.plot(model.virtualWealth[0], color='blue')
plt.plot(model.virtualWealth[1], color='orange')
plt.plot(model.virtualWealth[2], color='red')
plt.plot(model.virtualWealth[3], color='green')
plt.yscale('log')
plt.show()