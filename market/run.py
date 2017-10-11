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

    #max range (timesteps) equals 415 (830 days)
    for i in range(amountSteps):
        model.step()
        all_history[i] += model.globalPriceHistory[i]
    #show results for simulation j after 415 timesteps
    all_history_save['results%s' % j]= model.globalPriceHistory


with open('data/simulationPrices.pkl', 'wb') as f:
    pickle.dump(all_history_save, f, pickle.HIGHEST_PROTOCOL)



#learning agent learns



for h in range(amountSteps):
    all_history[h] = all_history[h]/amountRuns
plt.plot(all_history)
plt.show()

    #plt.plot(model.globalPriceHistory)
    #plt.show()


    # Store the results
        #for agent in model.schedule.agents:
        #    all_wealth.append(agent.wealth)

        #plt.hist(all_wealth, bins=range(max(all_wealth)+1))
        #plt.show()