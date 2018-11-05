class Option(object):
    def __init__(self):
        self.type = True
        self.color = True
        self.speed = True
        self.stay = True
        self.toll = False
        self.countTraffic = False

    def getOptionTuple(self):
        return (self.type,self.color,self.speed,self.stay)
