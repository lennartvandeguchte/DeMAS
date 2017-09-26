from mesa.model import Model
from mesa.time import RandomActivation
from agents import Trader, RandomTrader, ChartistTrader
import numpy
import math

class Market(Model):
    def __init__(self, N):
        self.globalPrice = 2
        self.globalPriceHistory = [self.globalPrice]
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
            if(numpy.random.rand()<0.3):
                a = ChartistTrader(i, self, wealth, bitcoin)
            else:
                a = RandomTrader(i, self, wealth, bitcoin)
            self.schedule.add(a)

    def getGlobalPrice(self):
        return self.globalPrice

    def setGlobalPrice(self, price):
        self.globalPrice = price


    def marketMigration(self):
        pass


    def checkExpiration(self):
        for order in self.sellOrderBook:
            if (order.expirationTime <= 0):
                self.sellOrderBook.remove(order)
            else:
                order.expirationTime -= 1

        for order in self.buyOrderBook:
            if (order.expirationTime <= 0):
                self.buyOrderBook.remove(order)
            else:
                order.expirationTime -= 1



    def orderLists(self):
        self.sellOrderBook.sort(key=lambda order: order.priceLimit)
#        for order in self.sellOrderBook:
#            print("sell")
#            print(order.priceLimit)

        self.buyOrderBook.sort(key=lambda order: order.priceLimit, reverse=True)
#        for order in self.buyOrderBook:
#            print("buy")
#            print(order.priceLimit)


    def resolveOrders(self):
        price = self.globalPrice

        while(self.buyOrderBook and self.sellOrderBook and self.sellOrderBook[0].priceLimit <= self.buyOrderBook[0].priceLimit):
            
            print("trade")

            #variables
            price = (self.sellOrderBook[0].priceLimit + self.buyOrderBook[0].priceLimit) / 2
            amount = min(self.sellOrderBook[0].amountBtc,self.buyOrderBook[0].amountBtc)
            
            if(self.globalPrice > price):
                print("decrease")

            #trade
            self.sellOrderBook[0].trader.wealth += amount * price
            self.sellOrderBook[0].trader.sellContract -= amount
            self.buyOrderBook[0].trader.investment -= amount * price
            self.buyOrderBook[0].trader.bitcoin += amount
            self.buyOrderBook[0].amountBtc -= amount
            self.sellOrderBook[0].amountBtc -= amount
            
            #removing fullfilled trades
            if(self.sellOrderBook[0].amountBtc == 0):
                self.sellOrderBook.remove(self.sellOrderBook[0])
            if(self.buyOrderBook[0].amountBtc == 0):
                self.buyOrderBook.remove(self.buyOrderBook[0])

        self.setGlobalPrice(price)
        self.globalPriceHistory.append(price)
        print(self.globalPrice)
        #print(self.sellOrderBook[0])
  
                        



    def step(self):
        '''Advance the model by one step.'''
        #agents perform action
        self.schedule.step()

        #new traders entering market and traders leaving market
        self.marketMigration()

        #check if there are orders that expired
        self.checkExpiration()

        #order lists according to buy and sell limits
        self.orderLists()
        
        #match buy and sell orders and execute them
        self.resolveOrders()

