class option(object):
	def __init__(self):
		self.type = True
		self.color = True
		self.speed = True
		self.stay = True
		self.outputVideoFPS = 60
		self.toll = False
		self.debug = False
		self.countTraffic = False

	def getTrueOption(self):
		count = 0
		if self.type:
			count +=1
		if self.color:
			count +=1
		if self.speed:
			count +=1
		if self.stay:
			count +=1
		return count * 10