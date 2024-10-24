
class SwitchInformation(object):
    def __init__(self, switchType, values, thresholds, numberPreviousStepsForThreshold, initialValues=None):
        """
        Params:
            - switchType (SwitchType): what property to switch on
            - values (tuple of two values): the values that can be set by this switch. The type must correspond to the property described by the switchType
            - thresholds (array of floats): a single or two values that determine the thresholds for switching on this property
            - numberPreviousStepsForThreshold (int): how many previous steps are used when computing the average of the previous local order
            - initialValues (numpy array) [optional]: the initial value chosen by every individual in accordance with the switchType and the value options
        """
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