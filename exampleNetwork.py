import services.ServicePreparation as ServicePreparation
import services.ServiceMetric as ServiceMetric
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceOrientations as ServiceOrientations
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceGeneral as ServiceGeneral
import services.ServiceNetwork as ServiceNetwork

import evaluators.EvaluatorMultiComp as EvaluatorMultiComp

from enums.EnumMetrics import Metrics
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType

import numpy as np
import time

domainSize = (50, 50)
radius = 20
switchType = SwitchType.K

imin = 1
imax = 2

filenames = ServiceGeneral.createListOfFilenamesForI(f"test_info_ordered_predator_tmax=3000", minI=imin, maxI=imax, fileTypeString="csv")
modelParamsDensity, simulationDataDensity, switchValues = ServiceSavedModel.loadModels(filenames, loadSwitchValues=True, switchTypes=[switchType], loadFromCsv=True)

times, positions, orientations = simulationDataDensity[0]
connections = ServiceNetwork.measureInformationTransferSpeedViaInformationTransferDistance(switchValues=switchValues, targetSwitchValue=1, eventStart=1000, positions=positions, domainSize=domainSize, radius=radius)