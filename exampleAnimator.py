import animator.AnimatorMatplotlib as AnimatorMatplotlib
import services.ServiceSavedModel as ServiceSavedModel
import animator.Animator2D as Animator2D

"""
--------------------------------------------------------------------------------
PURPOSE 
Loads a saved model and creates a video.
--------------------------------------------------------------------------------
"""

datafileLocation = ""
filename = "test"
modelParams, simulationData = ServiceSavedModel.loadModel(f"{datafileLocation}{filename}.json", loadSwitchValues=False)

# Initalise the animator
animator = AnimatorMatplotlib.MatplotlibAnimator(simulationData, (25, 25, 100))

# prepare the animator
preparedAnimator = animator.prepare(Animator2D.Animator2D(modelParams), frames=modelParams["tmax"])

preparedAnimator.saveAnimation(f"{filename}.mp4")

# Display Animation
#preparedAnimator.showAnimation()