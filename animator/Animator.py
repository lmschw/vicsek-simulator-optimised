from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import DefaultValues as dv



class Animator(object):
    """
    Animates the quiver plot for the Vicsek data
    """
    def prepareAnimation(self, matplotlibFigure, frames=100, frameInterval = 10):
        """
        Prepares the 2D animator object for animation.

        parameters:
        matplotlibFigure: Matplotlibs figure object.
        frameInterval -- The interval between two frames.
        frames -- The number of frames used in the animation.

        returns self
        """
        self._matplotlibFigure = matplotlibFigure
        self._frames = frames
        self._frameInterval = frameInterval

        return self

    def setSimulationData(self, simulationData, domainSize, colours=None):
        """
        Sets
        keyword arguments:
        simulationData -- The simulation data array.
        domainSize -- The tuple that represents the lenghts of the square domain in each dimension.

        return:
        self
        """        
        self._time, self._positions, self._orientations = simulationData
        self._domainSize = domainSize

        if colours is None:
            self._colours = len(self._time) * [len(self._positions[0]) * ['k']]
        else:
            self._colours = colours

        return self
    
    def setParameters(self, n, k, noise, radius):
        self._n = n
        self._k = k
        self._noise = noise
        self._radius = radius

    def setParams(self, modelParams):
        self._n = modelParams["n"]
        self._k = modelParams["k"]
        self._noise = modelParams["noise"]
        self._radius = modelParams["radius"]
        self._neighbourSelectionMechanism = modelParams["neighbourSelectionMechanism"]
        self._domainSize = modelParams["domainSize"]
        

    def showAnimation(self):
        """
        Shows the animation

        returns self
        """
        self._getAnimation()
        plt.show()
        
        return self
    
    def saveAnimation(self, filename, fpsVar=25, codecVar="avi"):
        """
        Saves the animation. Requires FFMPEG

        returns
        Animator
        """
        print("Saving commenced...")
        animation = self._getAnimation()
        animation.save(filename=filename, writer="ffmpeg")
        print("Saving completed.")
        #plt.close()
        return self
    
    def _getAnimation(self):
        return self.animation if 'animation' in self.__dict__ else self._generateAnimation()

    def _generateAnimation(self):
        """
        Generate the animation.
        
        returns
        animation object
        """
        self.animation = FuncAnimation(self._matplotlibFigure, self._animate, interval=self._frameInterval, frames = self._frames)

        return self.animation
