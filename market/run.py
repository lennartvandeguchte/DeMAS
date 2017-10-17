from model import *
import matplotlib.pyplot as plt
import numpy
import json
import pickle

# run.py
#all_wealth = []
amountSteps = 415
amountRuns = 1
all_history = numpy.zeros(415)
all_history_save = {}
save_simulations = False
train_learning_agent = False
include_learning_agent = True

#amount of simulations
for j in range(amountRuns):
    # create new model for a simulation
    model = Market(include_learning_agent)

    #max range (timesteps) equals 415 (830 days), this simulates the period between 01-01-2012 and 10-04-2014
    for i in range(amountSteps):
        model.step()
        all_history[i] += model.globalPriceHistory[i]
    #show results for simulation j after 415 timesteps
    all_history_save['results%s' % j]= model.globalPriceHistory


# Save simulated data to train and test the learning agent
if(save_simulations):   
    with open('data/simulationPricesTEST5.pkl', 'wb') as f:
        pickle.dump(all_history_save, f, pickle.HIGHEST_PROTOCOL)

# Learning agent trains and tests
# This file can also be runned seperately to tune parameters
if(train_learning_agent):   
    exec(open("trainLearningModel.py").read()) 


# Show the average simulated bitcoin price over an amount of runs
for h in range(amountSteps):
    all_history[h] = all_history[h]/amountRuns
plt.plot(all_history)
plt.show()

   


# Store the results
    #for agent in model.schedule.agents:
    #    all_wealth.append(agent.wealth)

    #plt.hist(all_wealth, bins=range(max(all_wealth)+1))
    #plt.show()