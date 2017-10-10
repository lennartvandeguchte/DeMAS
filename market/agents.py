from mesa import Agent
import numpy 
from order import Order
import random
from learningAgent import buyBitcoin

#from learningAgent import *


#agents parent class
class Trader(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, wealth, bitcoin):
        super().__init__(unique_id, model)
        self.wealth = wealth
        self.investment = 0
        self.bitcoin = bitcoin
        self.sellContract = 0
        self.keepTrading = True

    #behaviours of the agents
    #wants to buy
    def buy(self):
        globalPrice = self.model.getGlobalPrice()
        buyLimit = numpy.random.rand(1)[0] * self.wealth
        amountBtc = buyLimit / globalPrice
        mu = 1.02
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = globalPrice * g

        o = Order(self, amountBtc, priceLimit, self.expirationTime, 0)
        self.model.buyOrderBook.append(o) 

        self.wealth - buyLimit
        self.investment = buyLimit

    #wants to sell
    def sell(self):
        globalPrice = self.model.getGlobalPrice()
        amountBtc = numpy.random.rand(1)[0] * self.bitcoin
        mu = 1.00
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = globalPrice * g

        o = Order(self, amountBtc, priceLimit, self.expirationTime, 1)
        self.model.sellOrderBook.append(o)

        self.bitcoin - amountBtc
        self.sellContract = amountBtc

    #wants to retire
    def stopTrading(self):
        globalPrice = self.model.getGlobalPrice()

        if(self.bitcoin > 0):
            o = Order(self, self.bitcoin, globalPrice, 1, 1)
            self.model.sellOrderBook.append(o)

            self.sellContract = self.bitcoin
            self.bitcoin = 0

        elif(self.bitcoin == 0 and self.sellContract == 0):
            self.model.schedule.agents.remove(self)

        pass




#trader that makes random trades
class RandomTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 3
    
    #do behaviour for step at time t
    def step(self):
        if(self.keepTrading):
            if(numpy.random.rand()<=0.1):    
                if(numpy.random.rand()<=0.5):
                    self.buy()
                else:
                    self.sell()
            else:
                pass
        else:
            self.stopTrading()


#trader that buys when globalprice is increasing and sells if globalprice is decreasing
class ChartistTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 1
        self.lookBackTime = numpy.random.randint(30)+1

     #do behaviour for step at time t
    def step(self):
        if(self.keepTrading):
            if(numpy.random.rand()<=0.5 and len(self.model.globalPriceHistory) > self.lookBackTime):
                if(self.model.globalPriceHistory[-1] > (sum(self.model.globalPriceHistory[-self.lookBackTime:])/self.lookBackTime)):
                    self.buy()
                else:
                    self.sell()
            else: 
                pass
        else:
            self.stopTrading()



#trader that buys when globalprice is increasing and sells if globalprice is decreasing
class SelfLearningTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin, time):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 1
        self.time = time

    #do behaviour for step at time t
    def step(self):
        if(self.keepTrading):
            if(numpy.random.rand()<=0.5):
                action = buyBitcoin(self.model.learningModel, self.model.globalPriceHistory, self.time)
                if(action == 0):
                    self.buy()
                elif(action==1):
                    self.sell()
                else:
                    pass
            else: 
                pass
        else:
            self.stopTrading()