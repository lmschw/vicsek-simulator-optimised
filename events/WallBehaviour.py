import numpy as np

from enums.EnumWallInfluenceType import WallInfluenceType

import services.ServiceMetric as ServiceMetric
import services.ServiceOrientations as ServiceOrientations

# TODO refactor

class WallType(object):
    def __init__(self, name, wallInfluenceType, influenceDistance=None, focusPoint=[None, None], radius=None, cornerPoints=[None], areParticlesWithinWall=True):
        self.name = name
        self.wallInfluenceType = wallInfluenceType
        self.influenceDistance = influenceDistance
        self.focusPoint = focusPoint
        self.radius = radius
        self.cornerPoints = cornerPoints
        self.areParticlesWithinWall = areParticlesWithinWall

        self.checkCompleteness()

    def getParameterSummary(self):
        return f"{self.name}_wit={self.wallInfluenceType.value}_fp={self.focusPoint}_r={self.radius}_idist={self.influenceDistance}_apww={self.areParticlesWithinWall}"

    def checkClosenessToBorder(self, positions):
        isInside = self.isInsideOfWalls(positions) 
        sameSide = isInside == self.areParticlesWithinWall

        match(self.wallInfluenceType):
            case WallInfluenceType.FULL_AREA:
                return isInside & sameSide
            case WallInfluenceType.EXCEPT_NEAR_BORDER:
                return self.getDistanceFromBorder(positions) >= self.influenceDistance & sameSide
            case WallInfluenceType.CLOSE_TO_BORDER:
                return self.getDistanceFromBorder(positions) <= self.influenceDistance & sameSide
        
    def isInsideOfWalls(self, positions):
        pass

    def getDistanceFromBorder(self, positions):
        pass

    def getAvoidanceOrientation(self, orientations):
        pass

    def checkCompleteness(self):
        hasFocusPoint = not self.__isPointOrListNull(self.focusPoint)
        hasCornerPoints = not self.__isPointOrListNull(self.cornerPoints)

        if self.wallInfluenceType != WallInfluenceType.FULL_AREA and self.influenceDistance == None:
            raise Exception("If the wall event does not affect all particles within that area, an influenceDistance needs to be specified.")
        if hasFocusPoint and self.radius == None or self.radius != None and not hasFocusPoint:
            raise Exception("If a focus point or radius is supplied, so must the other value be.")
        if hasCornerPoints and (self.radius != None or hasFocusPoint):
            raise Exception("If corner points are provided, radius and focus point may not be supplied.")

    def __isPointOrListNull(self, pointOrList):
        return any(ele is None for ele in pointOrList)
    

class WallTypeCircle(WallType):
    def __init__(self, name, wallInfluenceType, focusPoint, radius, influenceDistance=None):
        super().__init__(name=name, wallInfluenceType=wallInfluenceType, influenceDistance=influenceDistance, focusPoint=focusPoint, radius=radius)
    
    def getParameterSummary(self):
        return f"{self.name}_wit={self.wallInfluenceType.value}_fp={self.focusPoint}_r={self.radius}_idist={self.influenceDistance}"

    def isInsideOfWalls(self, positions):
        return ((self.focusPoint[0] - positions[:, 0])**2 + (self.focusPoint[1] - positions[:, 1])**2) < self.radius **2 

    def getDistanceFromBorder(self, positions):
        return np.absolute((((self.focusPoint[0] - positions[:, 0])**2 + (self.focusPoint[1] - positions[:, 1])** 2)**(1/2)) - self.radius)
    
    def getAvoidanceOrientations(self, positions, orientations, speeds, dt, turnBy=0.3):
        if self.wallInfluenceType == WallInfluenceType.FULL_AREA:
            isAffected = np.full(len(positions), True)
        else:
            isAffected = self.getDistanceFromBorder(positions) >=self.influenceDistance
        
        closestPointOnCircle = self.getClosestPointToCircle(positions)
        anglesToClosestPoint = ServiceOrientations.computeAnglesWithRespectToFocusPoint(closestPointOnCircle, self.focusPoint)
        anglesOrientations = ServiceOrientations.computeAnglesForOrientations(orientations)
        isOrientationsAngleGreater = anglesOrientations > anglesToClosestPoint
        newOrientations = ServiceOrientations.computeUvCoordinatesForList(self.getAngleToAvoidCollision(positions, speeds, dt, anglesOrientations, isOrientationsAngleGreater, turnBy))
        isAffectedDoubled = np.column_stack([isAffected, isAffected])
        return np.where((isAffectedDoubled), newOrientations, orientations)

    def getClosestPointToCircle(self, positions):
        vX = positions[:, 0] - self.focusPoint[0]
        vY = positions[:, 1] - self.focusPoint[1]
        magV = np.sqrt(vX*vX + vY*vY)
        aX = self.focusPoint[0] + vX / magV * self.radius
        aY = self.focusPoint[1] + vY / magV * self.radius 
        return np.column_stack([aX, aY])
    
    def getAngleToAvoidCollision(self, positions, speeds, dt, angleOrientations, isOrientationsAngleGreater, turnBy=1):
        turns = (2*np.pi) / turnBy
        turn = 0
        orientations = ServiceOrientations.computeUvCoordinates(angleOrientations)
        willCollide = self.__willCollide(positions, orientations, dt, speeds)
        while np.count_nonzero(willCollide) > 0 and turn <= turns:
            angleOrientations = self.__turnAngle(angleOrientations, willCollide, isOrientationsAngleGreater, turnBy=turnBy)
            orientations = ServiceOrientations.computeUvCoordinates(angleOrientations)
            willCollide = self.__willCollide(positions, orientations, dt, speeds)
            turn += 1
        angleOrientations = np.where((willCollide) & (isOrientationsAngleGreater), angleOrientations + np.pi, angleOrientations)
        angleOrientations = np.where((willCollide) & (isOrientationsAngleGreater == False), angleOrientations - np.pi, angleOrientations)

        return angleOrientations
    
    def __turnAngle(self, angleOrientations, willCollide, isOrientationsAngleGreater, turnBy=0.03):
        angleOrientations = np.where((willCollide) & (isOrientationsAngleGreater), angleOrientations + turnBy, angleOrientations)
        angleOrientations = np.where((willCollide) & (isOrientationsAngleGreater == False), angleOrientations - turnBy, angleOrientations)
        return angleOrientations
    
    def __willCollide(self, positions, orientations, dt, speeds):
        do = dt * (speeds * orientations).T
        newPos = positions + do
        return self.isInsideOfWalls(positions) == self.isInsideOfWalls(newPos)
