import numpy as np

from enums.EnumSwitchType import SwitchType

class SwitchSummary(object):

    # TODO refactor to allow multiple instances of the same SwitchType

    def __init__(self, switches):
        """
        Params:
            - switches (list of SwitchInformation): all the switches active at any point in the simulation. only one SwitchInformation per type is allowed
        """
        self.rawSwitches = switches
        self.switches = self.__transformSwitches(switches)
        self.actives = self.__determineActives()

    def getParameterSummary(self):
        return "_".join([switch.getParameterSummary() for switch in self.rawSwitches])
    
    def isActive(self, switchType):
        """
        Determines whether a given SwitchType is active.

        Params:
            - switchType (SwitchType): which property is affected by the switch

        Returns:
            Whether the switch is active in the current simulation.
        """
        return self.actives[switchType]
    
    def getBySwitchType(self, switchType):
        """
        Retrieves the SwitchInformation for a given SwitchType.

        Params:
            - switchType (SwitchType): which property is affected by the switch

        Returns:
            The SwitchInformation for the SwitchType if present, None otherwise.
        """
        if self.isActive(switchType):
            return self.switches[switchType]
        return None
    
    def getMinMaxValuesForKSwitchIfPresent(self):
        """
        Determines the minimum and maximum value of k in the values of the SwitchType.K if it is present.

        Params:
            None

        Returns:
            Two integers representing the minimum and maximum value of k. Returns None, None if k-switching is not active.
        """
        if self.isActive(SwitchType.K):
            kSwitch = self.getBySwitchType(SwitchType.K)
            kMin = np.min([kSwitch.orderSwitchValue, kSwitch.disorderSwitchValue])
            kMax = np.max([kSwitch.orderSwitchValue, kSwitch.disorderSwitchValue]) 
            return kMin, kMax
        return None, None
    
    def __transformSwitches(self, switches):
        """
        Transforms the switches into a dictionary for ease of use.

        Params:
            - switches (list of SwitchInformation): the switches in the simulation

        Returns:
            A dictionary containing all switches with the SwitchType as its key and the SwitchInformation as its value
        """
        return {switch.switchType: switch for switch in switches}

    def __determineActives(self):
        """
        Determines which SwitchTypes are in the current simulation.

        Params:
            None

        Returns:
            A dictionary containing the SwitchType as its key and whether it is active as its value.
        """
        actives = {}
        for switchType in SwitchType:
            isPresent = False
            for switch in self.rawSwitches:
                if switch.switchType == switchType:
                    isPresent = True
            actives[switchType] = isPresent
        return actives
