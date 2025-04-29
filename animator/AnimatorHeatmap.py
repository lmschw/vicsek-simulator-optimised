from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


class Animator(object):

    """
    Animates the local order heatmap for the domain.
    """

    def prepareAnimation(self, matplotlibFigure=None, ax=None, title=None, valrange=(0, 1), frames=100, frameInterval = 10):
        """
        Prepares the animator object for the heatmap animation.

        parameters:
            - matplotlibFigure (Matplotlib Figure): Matplotlib figure object.
            - ax (array of Axes): all the relevant axes of the Matplotlib figure object
            - title (string) [optional]: title to be included in the animation
            - valrange (tuple of int: (min, max)) [optional]: the lower and upper limit of the y-axis
            - frames (int): The number of frames used in the animation.
            - frameInterval (int): The interval between two frames.

        returns self
        """
        if matplotlibFigure == None:
            fig = plt.figure(figsize=(8,8))
            self._ax = fig.add_subplot(autoscale_on = False)
            self._matplotlibFigure = fig   
        else:
            self._matplotlibFigure = matplotlibFigure
            self._ax = ax
        self._title = title
        self._valrange = valrange
        self._frames = frames
        self._frameInterval = frameInterval

        return self

    def setSimulationData(self, data):
        """
        Sets the simulation data
        
        Parameters:
            - simulationData: The simulation data array.
            - domainSize: The tuple that represents the lenghts of the square domain in each dimension.

        return:
        self
        """        
        self._time, self._values = data

        return self

    def showAnimation(self):
        """
        Shows the animation

        returns self
        """
        self._getAnimation()
        plt.show()
        
        return self
    
    def saveAnimation(self, filename):
        """
        Saves the animation. Requires FFMPEG

        returns
        Animator
        """
        print("Saving commenced...")
        animation = self._getAnimation()
        animation.save(filename=filename, writer="ffmpeg")
        print("Saving completed.")
        return self
    
    def _getAnimation(self):
        return self.animation if 'animation' in self.__dict__ else self._generateAnimation()

    def _generateAnimation(self):
        """
        Generate the animation.
        
        returns
        animation object
        """ 
        self.plot = plt.matshow(self._values[0], fignum=0)
        colourbar = self._matplotlibFigure.colorbar(self.plot)
        colourbar.ax.set_ylim(self._valrange)
        #colourbar.norm.autoscale(self._valrange)

        self.animation = FuncAnimation(self._matplotlibFigure, self._animate, interval=self._frameInterval, frames = self._frames)
        return self.animation
    

    def init(self):
        self.plot.set_data(self._values[0])
        return self.plot
    
    def _animate(self, i):
        """
        Animator class that goes through sim data.

        keyword arguments:
        i -- Loop index.
        """
        self.plot.set_data(self._values[i])
        if self._title == None:
            plt.title(f"$t$={self._time[i]:.2f}")
        else:
            plt.title(f"{self._title}\n$t$={self._time[i]:.2f}")

        return [self.plot]
