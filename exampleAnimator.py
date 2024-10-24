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
filename = "test"
modelParams, simulationData, switchValues, colours = ServiceSavedModel.loadModel(f"{datafileLocation}{filename}.json", loadSwitchValues=True, loadColours=True)

# Initalise the animator
animator = AnimatorMatplotlib.MatplotlibAnimator(simulationData, (25, 25, 100), colours=colours)

# prepare the animator
preparedAnimator = animator.prepare(Animator2D(modelParams), frames=modelParams["tmax"])
#preparedAnimator = animator.prepare(Animator2D(modelParams), frames=100)

preparedAnimator.saveAnimation(f"{filename}.mp4")

# Display Animation
#preparedAnimator.showAnimation()