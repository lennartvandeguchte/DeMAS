from model import *
import matplotlib.pyplot as plt

# run.py
#all_wealth = []

#amount of simulations
for j in range(1):
    # create new model for a simulation
    model = Market()

    #max range (timesteps) equals 415 (830 days)
    for i in range(415):
        model.step()

    #show results for simulation j after 415 timesteps
    plt.plot(model.globalPriceHistory)
    plt.show()


    # Store the results
        #for agent in model.schedule.agents:
        #    all_wealth.append(agent.wealth)

        #plt.hist(all_wealth, bins=range(max(all_wealth)+1))
        #plt.show()