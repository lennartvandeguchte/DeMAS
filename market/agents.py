from mesa import Agent
import numpy 
from order import Order
import random



class Trader(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, wealth, bitcoin):
        super().__init__(unique_id, model)
        self.wealth = wealth
        self.bitcoin = bitcoin
        self.model = model
    """
    def step(self):
        # The agent's step will go here.
        if self.wealth <= 10:
            return
        other_agent = random.choice(self.model.schedule.agents)
        other_agent.wealth += 1
        self.wealth -= 1
        pass

        """
		


class RandomTrader(Trader):
    expirationTime = 1


    def buy(self):
        #buy
        buyLimit = numpy.random.rand(1) * self.wealth
        amountBtc = buyLimit * model.globalPrice
        mu = 1.02
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = globalPrice * g

        o = Order(amountBtc, priceLimit, expirationTime, self, 0)
        model.buyOrderBook.append(o)    

    def sell(self):
        #sell
        amountBtc = numpy.random.rand(1) * self.bitcoin
        mu = 1.00
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = model.globalPrice * g

        o = Order(amountBtc, priceLimit, expirationTime, self, 1)
        model.sellOrderBook.append(o)

    def step(self):
        if(numpy.random.randint(2)<1):
            buy()
        else:
            sell()
            





class ChartistTrader(Trader):
    def step(self):
        pass