import matplotlib.pyplot as plt

class MatplotlibAnimator:
    """
    The animator instance driven by Matplotlib.
    """
    
    def __init__(self, simulationData, domainSize, colours=None):
        """
        Constructor.

        keyword arguments:
        simulationData -- The simulation data array.
        domainSize -- The tuple that represents the lenghts of the square domain in each dimension.
        """
        self._simulationData = simulationData
        self._domainSize = domainSize
        self._colours = colours

        self._initialize()

    def prepare(self, animator, frames=100, frameInterval=10):
        """
        Prepares the appropriate animator.

        keyword arguments:
        animator -- The appropriate animator class.

        return:
        Prepared animator feeded with simulation data.
        """
        preparedAnimator =  animator.prepareAnimation(self._figure, frames, frameInterval)

        return preparedAnimator.setSimulationData(self._simulationData, self._domainSize, self._colours)

    def _initialize(self):
        """
        Initializes matplotlib for animation.
        
        return:
        plt.figure()
        """
        self._figure = plt.figure()
        
        return self._figure