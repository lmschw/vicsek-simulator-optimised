import ServicePreparation
import ServiceMetric
import ServiceVicsekHelper
import ServiceOrientations
import ServiceSavedModel

import numpy as np



#print(ServicePreparation.getNumberOfParticlesForConstantDensity(0.1, (100, 100)))
#print(ServicePreparation.getDensity((100, 100), 100))
#print(ServicePreparation.getDomainSizeForConstantDensity(0.01, 5))

switchValues = [5,1]
domainSize = (100, 100)
n = 10
radius = 20


"""

positions = domainSize*np.random.rand(n,len(domainSize))
orientations = ServiceOrientations.normalizeOrientations(np.random.rand(n, len(domainSize))-0.5)

orientations[1] = orientations[8]
orientations[3] = orientations[8]
orientations[7] = orientations[1]
orientations[0] = orientations[2]



initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(switchValues[0], domainSize, n, angleX=0.5, angleY=0.5)
positions = initialState[0]
orientations = initialState[1]


print(ServiceMetric.findClustersWithRadius(positions, orientations, domainSize, radius, threshold=0.01))
"""

modelParams, simulationData = ServiceSavedModel.loadModel("test.json")
time, positions, orientations = simulationData

for t in [0, 1000, 2000, 3000]:
    print(ServiceMetric.findClustersWithRadius(positions[t], orientations[t], domainSize, radius, threshold=0.01))