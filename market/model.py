from mesa.model import Model
from mesa.time import RandomActivation
from agents import Trader, RandomTrader
import numpy

class Market(Model):
    def __init__(self, N):
        self.globalPrice = 2
        self.num_agents = N
        self.schedule = RandomActivation(self)
        sellOrderBook = []
        buyOrderBook = []
        # Create agents
        for i in range(self.num_agents):
            wealth = numpy.random.randint(10)
            bitcoin = 1

            a = RandomTrader(i, self, wealth, bitcoin)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
        print (sellOrderBook)