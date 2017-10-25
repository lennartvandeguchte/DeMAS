from mesa import Agent
import numpy 
from order import Order
import random
from functionsLearningAgent import buySellOrPass

# Agents parent class
class Trader(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, wealth, bitcoin):
        super().__init__(unique_id, model)
        self.wealth = wealth
        self.investment = 0
        self.bitcoin = bitcoin
        self.sellContract = 0
        self.keepTrading = True

    # Behaviours of the agents
    # Wants to buy
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
        self.investment = buyLimit

    # Wants to sell
    def sell(self):
        globalPrice = self.model.getGlobalPrice()
        amountBtc = numpy.random.rand(1)[0] * self.bitcoin
        mu = 1.00
        sigma = 0.05    #todo dependent standard deviation global price
        g = numpy.random.normal(mu, sigma)
        priceLimit = globalPrice * g

        o = Order(self, amountBtc, priceLimit, self.expirationTime, 1)
        self.model.sellOrderBook.append(o) 
        self.sellContract = amountBtc

    # Wants to retire
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



# Trader that makes random trades
class RandomTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 3
    
    # Do behaviour for step at time t
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


# Trader that buys when globalprice is increasing and sells if globalprice is decreasing
class ChartistTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 1
        #self.lookBackTime = numpy.random.randint(30)+1
        self.lookBackTime = 10


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


# Trader that performs a behaviour according to what it has learning in the past
class SelfLearningTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 3
        self.actionHistory = []

    # Do behaviour for step at time t
    def step(self):
        action = buySellOrPass(self.model.globalPriceHistory, self.model.learningModel)
        print('action', action)
        self.actionHistory = numpy.append(self.actionHistory, action)
        print('wealth LA', self.wealth)
        print('bitcoin LA', self.bitcoin)
        if(action == 1):
            self.buy()
        elif(action==2):
            self.sell()
        else:
            pass


# Trader that buys when a local minimum is reached and sell when a local maximum is reached
class FastTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 1

    # Do behaviour for step at time t
    def step(self):
        if((self.model.schedule.time > 1) and (self.model.globalPriceHistory[-2] < self.model.getGlobalPrice()) and (self.model.globalPriceHistory[-3] > self.model.globalPriceHistory[-2])):
            self.buy()
            print('fast trader buys')
        elif((self.model.schedule.time > 1) and (self.model.globalPriceHistory[-2] > self.model.getGlobalPrice()) and (self.model.globalPriceHistory[-3] < self.model.globalPriceHistory[-2])):  
            self.sell()
            print('fast trader sells')
        else: 
            pass


