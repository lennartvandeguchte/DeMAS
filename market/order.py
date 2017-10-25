
# Creates an order
class Order:
	def __init__(self, trader, amountBtc,priceLimit, expirationTime, buyOrSell):
		self.trader = trader
		self.amountBtc = amountBtc
		self.priceLimit = priceLimit
		self.expirationTime = expirationTime
		self.buyOrSell = buyOrSell


	def __str__(self):
		return  str(self.trader) + " wants to " + str(self.buyOrSell) + " " + str(self.amountBtc) + " bitcoins for " + str(self.priceLimit)