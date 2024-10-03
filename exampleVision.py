import numpy as np

import services.ServiceVision as ServiceVision
import services.ServicePreparation as ServicePreparation

import animator.Animator

positions, orientations = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(startSwitchTypeValue=None, domainSize=(25, 25),
                                                                                                     numberOfParticles=10, angleX=0.5, angleY=0.5)




"""
angles = [-np.pi, -(1/2) * np.pi, 0, (1/3)* np.pi, (2/3)* np.pi, np.pi, (4/3) * np.pi, (5/3) * np.pi, 2 * np.pi, (7/3)*np.pi, 3*np.pi]
newAngles = ServiceVision.normaliseAngles(angles)
for i in range(len(angles)):
    print(f"old: {angles[i]}, new: {newAngles[i]}, old+2pi: {angles[i] + (2*np.pi)}, old%2pi: {angles[i] % (2*np.pi)}")
"""
print(positions)
minAngles, maxAngles = ServiceVision.determineMinMaxAngleOfVision(orientations=orientations, degreesOfVision=(5/6)*np.pi)
print(f"pos 0: {positions[0]}")
print(f"minangles 0:{minAngles[0]}")
print(f"minangles 0:{maxAngles[0]}")

isInFov = ServiceVision.isInFieldOfVision(positions, minAngles, maxAngles)
print(f"isInFov 0: {isInFov}")
