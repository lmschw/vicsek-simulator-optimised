
class SwitchInformation(object):
    def __init__(self, switchType, values, thresholds, numberPreviousStepsForThreshold, initialValues=None):
        self.switchType = switchType
        self.thresholds = thresholds
        self.orderSwitchValue, self.disorderSwitchValue = values
        self.numberPreviousStepsForThreshold = numberPreviousStepsForThreshold
        self.initialValues = initialValues

        self.lowerThreshold, self.upperThreshold = self.__getLowerAndUpperThreshold()

    def getParameterSummary(self):
        return f"{self.switchType.value}_o={self.orderSwitchValue}_do={self.disorderSwitchValue}_th={self.lowerThreshold}-{self.upperThreshold}_p={self.numberPreviousStepsForThreshold}"

    def __getLowerAndUpperThreshold(self):
        """
        Determines the lower and upper thresholds for hysteresis.

        Params:
            None

        Returns:
            Two floats representing the lower and upper threshold respectively
        """
        if len(self.thresholds) == 1:
            switchDifferenceThresholdLower = self.thresholds[0]
            switchDifferenceThresholdUpper = 1 - self.thresholds[0]
        else:
            switchDifferenceThresholdLower = self.thresholds[0]
            switchDifferenceThresholdUpper = self.thresholds[1]
        return switchDifferenceThresholdLower, switchDifferenceThresholdUpper