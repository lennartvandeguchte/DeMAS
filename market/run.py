from model import *
import matplotlib.pyplot as plt
import numpy

# run.py
#all_wealth = []
#all_history = []
all_history = numpy.zeros(415)

#amount of simulations
for j in range(100):
    # create new model for a simulation
    model = Market()

    #max range (timesteps) equals 415 (830 days)
    for i in range(415):
        model.step()
        all_history[i] += model.globalPriceHistory[i]
    #show results for simulation j after 415 timesteps

for h in range(100):
    all_history[i] = all_history[i]/100
plt.plot(all_history)
plt.show()

    #plt.plot(model.globalPriceHistory)
    #plt.show()


    # Store the results
        #for agent in model.schedule.agents:
        #    all_wealth.append(agent.wealth)

        #plt.hist(all_wealth, bins=range(max(all_wealth)+1))
        #plt.show()