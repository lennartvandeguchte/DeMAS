from mesa import Agent
import numpy 
from order import Order
import random



class Trader(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, wealth, bitcoin):
        super().__init__(unique_id, model)
        self.wealth = wealth
        self.investment = 0
        self.bitcoin = bitcoin
        self.sellContract = 0
        self.keepTrading = True
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


    def stopTrading(self):
        globalPrice = self.model.getGlobalPrice()

        if(self.bitcoin > 0):
            o = Order(self, self.bitcoin, globalPrice, 1, 1)
            self.model.sellOrderBook.append(o)

            self.sellContract = self.bitcoin
            self.bitcoin = 0
            print("stopped trading")
        elif(self.bitcoin == 0 and self.sellContract == 0):
            self.model.schedule.agents.remove(self)
            print("removed self")
        pass





class RandomTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 3
    

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



class ChartistTrader(Trader):
    def __init__(self, unique_id, model, wealth, bitcoin):
        Trader.__init__(self, unique_id, model, wealth, bitcoin)
        self.expirationTime = 1


    def step(self):
        if(self.keepTrading):
            if(numpy.random.rand()<=0.5):
                if(self.model.globalPriceHistory[-1] > (sum(self.model.globalPriceHistory)/len(self.model.globalPriceHistory))):
                    self.buy()
                    print("chartist buy")
                else:
                    self.sell()
                    print("chartist sell")
            else: 
                pass
        else:
            self.stopTrading()