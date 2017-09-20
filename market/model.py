from mesa.model import Model
from mesa.time import RandomActivation
from agents import Trader, RandomTrader, ChartistTrader
import numpy
import math

class Market(Model):
    def __init__(self, N):
        self.globalPrice = 2
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.sellOrderBook = []
        self.buyOrderBook = []

        # Create agents
        for i in range(self.num_agents):
            wealth = numpy.random.pareto(0.6) * 100
            wealth = math.floor(wealth)
            bitcoin = 1
            print(wealth)
            a = RandomTrader(i, self, wealth, bitcoin)
            self.schedule.add(a)

    def getGlobalPrice(self):
        return self.globalPrice

    def setGlobalPrice(self, price):
        self.globalPrice = price


    def checkExpiration(self):
        pass


    def orderLists(self):
        self.sellOrderBook.sort(key=lambda order: order.priceLimit, reverse=True)
#        for order in self.sellOrderBook:
#            print("sell")
#            print(order.priceLimit)

        self.buyOrderBook.sort(key=lambda order: order.priceLimit)
#        for order in self.buyOrderBook:
#            print("buy")
#            print(order.priceLimit)


    def resolveOrders(self):
        price = self.globalPrice

        while(self.buyOrderBook and self.sellOrderBook and self.sellOrderBook[0].priceLimit <= self.buyOrderBook[0].priceLimit):
            
            #variables
            price = (self.sellOrderBook[0].priceLimit + self.buyOrderBook[0].priceLimit) / 2
            amount = min(self.sellOrderBook[0].amountBtc,self.buyOrderBook[0].amountBtc)
            
            #trade
            self.sellOrderBook[0].trader.wealth += amount * price
            self.sellOrderBook[0].trader.bitcoin -= amount
            self.buyOrderBook[0].trader.wealth -= amount * price
            self.buyOrderBook[0].trader.bitcoin += amount
            self.buyOrderBook[0].amountBtc -= amount
            self.sellOrderBook[0].amountBtc -= amount
            
            #removing fullfilled trades
            if(self.sellOrderBook[0].amountBtc == 0):
                self.sellOrderBook.remove(self.sellOrderBook[0])
            if(self.buyOrderBook[0].amountBtc == 0):
                self.buyOrderBook.remove(self.buyOrderBook[0])

        self.setGlobalPrice(price)
        print(self.globalPrice)
        #print(self.sellOrderBook[0])
  
                        



    def step(self):
        '''Advance the model by one step.'''
        #agents perform action
        self.schedule.step()

        #check if there are orders that expired
        self.checkExpiration()

        #order lists according to buy and sell limits
        self.orderLists()
        
        #match buy and sell orders and execute them
        self.resolveOrders()

