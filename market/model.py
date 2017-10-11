from mesa.model import Model
from mesa.time import RandomActivation
from agents import Trader, RandomTrader, ChartistTrader#, SelfLearningTrader
from keras.models import model_from_json
import pandas as pd
import numpy
import math
import random
import sys

class Market(Model):
    def __init__(self):
        self.globalPrice = 5.5
        self.globalPriceHistory = [self.globalPrice] #self.loadBitcoinData(200)
        print(self.globalPriceHistory)
        self.num_agents = 0
        self.num_agents_historical = numpy.genfromtxt('./inputs/n-unique-addresses.csv', delimiter=',')
        self.num_bitcoins = 789
        self.num_bitcoins_historical = 789
        self.schedule = RandomActivation(self)
        self.sellOrderBook = []
        self.buyOrderBook = []

        self.num_agents = math.floor(self.num_agents_historical[0]/100)
        print(self.num_agents)

        # load and create learning model
        json_file = open('inputs\learningModel.json', 'r')
        loadedModel = json_file.read()
        json_file.close()
        self.learningModel = model_from_json(loadedModel)
        # load weights into new model
        self.learningModel.load_weights("inputs\model.h5")
        print("Loaded model from disk")

        # Create agents
        for i in range(self.num_agents):
            wealth = numpy.random.pareto(0.6) * 100
            wealth = math.floor(wealth)
            bitcoin = 1
            #print(wealth)
            #if(i==1 and ): ### One self learning agent
            #    a = SelfLearningTrader(i, self, wealth, bitcoin, self.schedule.time)
            if(numpy.random.rand()<0.3):
                a = ChartistTrader(i, self, wealth, bitcoin)
            else:
                a = RandomTrader(i, self, wealth, bitcoin)
            self.schedule.add(a)

    def loadBitcoinData(self, numberOfDays):
        prices = pd.read_pickle('data/BITSTAMP_1day.pkl')
        prices = prices.iloc[(-2102-numberOfDays):-2102,]
        return prices['Close'].values


    #setters and getters
    def getGlobalPrice(self):
        return self.globalPrice

    def setGlobalPrice(self, price):
        self.globalPrice = price

    def getHistoricalBitcoins(self):
        return self.num_bitcoins_historical

    def setHistoricalBitcoins(self):
        t = self.schedule.time
        self.num_bitcoins_historical =math.floor(((4.709*10**-5)*(t**3)-(0.08932*(t**2))+98.88*t+78880)/100)  

    #functions
    def marketMigration(self):
        # checking for difference between data and model
        if(self.schedule.time < 415):
            historicalDifference = math.floor(self.num_agents_historical[math.floor(self.schedule.time/2)]/100) - self.num_agents
        else:
            historicalDifference = 0

        # there are not enough traders in the model   
        if(historicalDifference > 0):
            for i in range(historicalDifference):
                wealth = numpy.random.pareto(0.6) * 100
                wealth = math.floor(wealth)
                bitcoin = 0
                #print(wealth)
                if(numpy.random.rand()<0.3):
                    a = ChartistTrader(i, self, wealth, bitcoin)
                else:
                    a = RandomTrader(i, self, wealth, bitcoin)
                self.num_agents += 1
                self.schedule.add(a)
        elif(historicalDifference < 0):
            for i in range(historicalDifference*-1):
                self.num_agents -= 1
                size = len(self.schedule.agents) - 1
                randomPick = numpy.random.randint(0, size)
                while(self.schedule.agents[randomPick].keepTrading != True):
                    randomPick = numpy.random.randint(0, size)
                self.schedule.agents[randomPick].keepTrading = False    
        else:
            pass

    
    def mining(self):
        self.setHistoricalBitcoins()
        historicalBitcoinDifference = self.num_bitcoins_historical - self.num_bitcoins
        if(historicalBitcoinDifference > 0):
            for i in range(historicalBitcoinDifference):
                size = len(self.schedule.agents) - 1
                randomPick = numpy.random.randint(0, size)
                while(type(self.schedule.agents[randomPick]) == RandomTrader):
                    randomPick = numpy.random.randint(0, size)
                self.schedule.agents[randomPick].bitcoin += 1
        else:
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
            
            #variables
            price = (self.sellOrderBook[0].priceLimit + self.buyOrderBook[0].priceLimit) / 2
            amount = min(self.sellOrderBook[0].amountBtc,self.buyOrderBook[0].amountBtc)


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
        self.globalPriceHistory = numpy.append(self.globalPriceHistory, price)
        print(self.globalPrice)
        #print(self.sellOrderBook[0])
  
                        


    #timestep function calls
    def step(self):
        '''Advance the model by one step.'''
        #agents perform action
        self.schedule.step()

        #new traders entering market and traders leaving market
        self.marketMigration()

        #mining new bitcoins
        self.mining()

        #check if there are orders that expired
        self.checkExpiration()

        #order lists according to buy and sell limits
        self.orderLists()
        
        #match buy and sell orders and execute them
        self.resolveOrders()

