import animator.AnimatorMatplotlib as AnimatorMatplotlib
import services.ServiceSavedModel as ServiceSavedModel
from animator.Animator2D import Animator2D

"""
--------------------------------------------------------------------------------
PURPOSE 
Loads a saved model and creates a video.
--------------------------------------------------------------------------------
"""

datafileLocation = ""
filename = "test_stress_1"
modelParams, simulationData = ServiceSavedModel.loadModel(f"{datafileLocation}{filename}.json", loadSwitchValues=False, loadColours=False)

# Initalise the animator
animator = AnimatorMatplotlib.MatplotlibAnimator(simulationData, (25, 25, 100),showRadiusForExample=False)

# prepare the animator
#preparedAnimator = animator.prepare(Animator2D(modelParams), frames=modelParams["tmax"])
preparedAnimator = animator.prepare(Animator2D(modelParams), frames=5000)

preparedAnimator.saveAnimation(f"{filename}_stress.mp4")

# Display Animation
#preparedAnimator.showAnimation()