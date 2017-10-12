from model import *
import matplotlib.pyplot as plt
import numpy
import json
import pickle

# run.py
#all_wealth = []
amountSteps = 415
amountRuns = 3
all_history = numpy.zeros(415)
all_history_save = {}

#amount of simulations
for j in range(amountRuns):
    # create new model for a simulation
    model = Market()

    #max range (timesteps) equals 415 (830 days), this simulates the period between 01-01-2012 and 10-04-2014
    for i in range(amountSteps):
        model.step()
        all_history[i] += model.globalPriceHistory[i]
    #show results for simulation j after 415 timesteps
    all_history_save['results%s' % j]= model.globalPriceHistory


# Save simulated data to train and test the learning agent
with open('data/simulationPrices.pkl', 'wb') as f:
    pickle.dump(all_history_save, f, pickle.HIGHEST_PROTOCOL)



# Learning agent trains and tests
# Can be commented and runned seperately to use the same simulated data, useful for parameter optimalization
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