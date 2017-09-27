from model import *
import matplotlib.pyplot as plt

# run.py
all_wealth = []
for j in range(1):
    # Run the model
    model = Market()
    #max range equals 415
    for i in range(415):
        model.step()

    plt.plot(model.globalPriceHistory)
    plt.show()


    # Store the results
        #for agent in model.schedule.agents:
        #    all_wealth.append(agent.wealth)

        #plt.hist(all_wealth, bins=range(max(all_wealth)+1))
        #plt.show()