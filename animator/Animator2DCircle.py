import matplotlib.pyplot as plt
import matplotlib.patches as patches
import animator.Animator as Animator

class Animator2D(Animator.Animator):
    """
    Animator class for 2D graphical representation for a circular domain
    """

    def __init__(self, modelParams):
        self.setParams(modelParams)

    def _animate(self, i):
        """
        Animator class that goes through sim data.

        Parameters:
            - i (int): Loop index.

        Returns:
            Nothing
        """

        if i % 500 == 0:
            print(i)

        plt.clf()
        plt.gca().add_patch(patches.Circle((25, 25), 10))
        plt.quiver(self._positions[i,:,0],self._positions[i,:,1],self._orientations[i,:,0],self._orientations[i,:,1],color=self._colours[i])
        plt.xlim(0,self._domainSize[0])
        plt.ylim(0,self._domainSize[1])
        plt.gca().set_aspect('equal')
        plt.title(f"$t$={self._time[i]:.2f}")