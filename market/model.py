from mesa.model import Model
from mesa.time import RandomActivation
from agents import MoneyAgent

class Market(Model):
    def __init__(self, N):

        self.num_agents = N
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()