import numpy as np
import matplotlib.pyplot as plt

import services.ServiceMetric as sm

class EvaluatorAvalanches:

    def __init__(self, orientations, orderThreshold=0.8, savePath=None):
        self.orientations = orientations
        self.orderThreshold = orderThreshold
        self.savePath = savePath

    def evaluateAvalanches(self):
        avalanches = self.measureAvalanches()
        n = len(avalanches)
        durationMin = np.min(avalanches[:,2])
        durationAvg = np.average(avalanches[:,2])
        durationMax = np.max(avalanches[:,2])
        magnitudeMin = self.orderThreshold-np.max(avalanches[:,4])
        magnitudeAvg = self.orderThreshold-np.average(avalanches[:,4])
        magnitudeMax = self.orderThreshold-np.min(avalanches[:,4])

        print(f"Number of avalanches: {n}")
        print(f"Duration avalanches: min={durationMin}, avg={durationAvg}, max={durationMax}")
        print(f"Magnitude: min={magnitudeMin}, avg={magnitudeAvg}, max={magnitudeMax}")

        self.plotProbabilityDistributionDurations(avalanches[:,2])
        return avalanches, n, (durationMin, durationAvg, durationMax), (magnitudeMin, magnitudeAvg, magnitudeMax)

    def measureAvalanches(self):
        avalanches = []
        startAvalanche = -1
        lowpointTimestep = -1
        lowpoint = -1
        for t in range(len(self.orientations)):
            localOrder = sm.computeGlobalOrder(orientations=self.orientations[t])
            if localOrder >= self.orderThreshold and startAvalanche != -1:
                avalanches.append([startAvalanche, t, t-startAvalanche, lowpointTimestep, lowpoint])
                startAvalanche = -1
                lowpointTimestep = -1
                lowpoint = -1
            elif localOrder < self.orderThreshold:
                if startAvalanche == -1:
                    startAvalanche = t
                    lowpointTimestep = t
                    lowpoint = localOrder
                elif localOrder < lowpointTimestep:
                    lowpointTimestep = t
                    lowpoint = localOrder
        return np.array(avalanches)

    def plotProbabilityDistributionDurations(self, durations):        
        # Plot histogram
        plt.hist(durations, bins=30, density=True, alpha=0.6, color='b')

        plt.xlabel("Duration")
        plt.ylabel("Probability Density")
        plt.title("Histogram of durations")

        if self.savePath:
            plt.savefig(f"{self.savePath}.pdf")
            plt.savefig(f"{self.savePath}.jpeg")

        plt.show()



    
