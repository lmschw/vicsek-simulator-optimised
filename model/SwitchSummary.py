import numpy as np

from enums.EnumSwitchType import SwitchType

class SwitchSummary(object):

    def __init__(self, switches):
        self.rawSwitches = switches
        self.switches = self.__transformSwitches(switches)
        self.actives = self.__determineActives()

    def getParameterSummary(self):
        return "_".join([switch.getParameterSummary() for switch in self.rawSwitches])
    
    def isActive(self, switchType):
        return self.actives[switchType]
    
    def getBySwitchType(self, switchType):
        if self.isActive(switchType):
            return self.switches[switchType]
        return None
    
    def getMinMaxValuesForKSwitchIfPresent(self):
        if self.isActive(SwitchType.K):
            kSwitch = self.getBySwitchType(SwitchType.K)
            kMin = np.min([kSwitch.orderSwitchValue, kSwitch.disorderSwitchValue])
            kMax = np.max([kSwitch.orderSwitchValue, kSwitch.disorderSwitchValue]) 
            return kMin, kMax
        return None, None
    
    def __transformSwitches(self, switches):
        return {switch.switchType: switch for switch in switches}

    def __determineActives(self):
        actives = {}
        for switchType in SwitchType:
            isPresent = False
            for switch in self.rawSwitches:
                if switch.switchType == switchType:
                    isPresent = True
            actives[switchType] = isPresent
        return actives
