import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Animator

class Animator2D(Animator.Animator):
    """
    Animator class for 2D graphical representation.
    """

    def __init__(self):
        """
        Constructor. Returns the Animator2D instance.
        """
        self.setParameters()

    def _animate(self, i):
        """
        Animator class that goes through sim data.

        keyword arguments:
        i -- Loop index.
        """

        plt.clf()
        plt.gca().add_patch(patches.Circle((20, 20), 10))
        plt.quiver(self._positions[i,:,0],self._positions[i,:,1],self._orientations[i,:,0],self._orientations[i,:,1],color=self._colours[i])
        plt.xlim(0,self._domainSize[0])
        plt.ylim(0,self._domainSize[1])
        plt.gca().set_aspect('equal')
        plt.title(f"$t$={self._time[i]:.2f}")