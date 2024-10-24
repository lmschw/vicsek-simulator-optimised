import matplotlib.pyplot as plt

class MatplotlibAnimator:
    """
    The animator instance driven by Matplotlib.
    """
    
    def __init__(self, simulationData, domainSize, colours=None, redIndices=[], showRadiusForExample=False):
        """
        Parameters:
            - simulationData (array of arrays): The simulation data array containing times, positions and orientations
            - domainSize (tuple of floats): The tuple that represents the lenghts of the square domain in each dimension.
            - colours (array of arrays) [optional]: Contains the colour for every particle at every timestep. By default, all particles will be shown in black
            - redIndices (list) [optional]: A list containing indices of particles that will be shown in red. Will be ignored if a colour array is passed
            - showRadiusForExample (boolean) [optional]: if an example particle is provided in the modelParams, toggles if the perception radius is drawn in the video
        """
        self._simulationData = simulationData
        self._domainSize = domainSize
        self._colours = colours
        self._redIndices = redIndices
        self._showRadiusForExample = showRadiusForExample

        self._initialize()

    def prepare(self, animator, frames=100, frameInterval=10):
        """
        Prepares the appropriate animator.

        Parameters:
            - animator (Animator): The appropriate animator class.

        Returns:
            Prepared animator feeded with simulation data.
        """
        preparedAnimator =  animator.prepareAnimation(self._figure, frames, frameInterval)

        return preparedAnimator.setSimulationData(self._simulationData, self._domainSize, self._colours, self._redIndices, self._showRadiusForExample)

    def _initialize(self):
        """
        Initializes matplotlib for animation.

        Parameters:
        None
        
        Returns:
            plt.figure()
        """
        self._figure = plt.figure()
        
        return self._figure