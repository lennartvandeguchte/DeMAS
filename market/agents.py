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
        #self.model = model
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
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 1
    
    

    def buy(self):
        globalPrice = self.model.getGlobalPrice()
        buyLimit = numpy.random.rand(1)[0] * self.wealth
        amountBtc = buyLimit / globalPrice
        mu = 1.50
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = globalPrice * g

        o = Order(self, amountBtc, priceLimit, self.expirationTime, 0)
        self.model.buyOrderBook.append(o)    


    def sell(self):
        globalPrice = self.model.getGlobalPrice()
        amountBtc = numpy.random.rand(1)[0] * self.bitcoin
        mu = 1.00
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = globalPrice * g

        o = Order(self, amountBtc, priceLimit, self.expirationTime, 1)
        self.model.sellOrderBook.append(o)


    def step(self):
        if(numpy.random.randint(2)<1):
            self.buy()
        else:
            self.sell()



class ChartistTrader(Trader):
    def step(self):
        pass