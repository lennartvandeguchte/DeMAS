from model import *
import matplotlib.pyplot as plt

# run.py
all_wealth = []
for j in range(100):
    # Run the model
    model = Market(10)
    for i in range(10):
        model.step()

    # Store the results
    for agent in model.schedule.agents:
        all_wealth.append(agent.wealth)

#plt.hist(all_wealth, bins=range(max(all_wealth)+1))