import numpy as np
import os

import model.ActiveElastic as ae
import services.ServiceSavedModel as ssm
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

# environment and simulation parameters
n_agents = 3
n_steps = 100
env_size = 100

# perception parameters
degrees_of_vision = 2*np.pi
radius = np.inf

# neighbour selection parameters
nsm = NeighbourSelectionMechanism.NEAREST
k = 5

# visualisation parameters
graph_freq = 10
visualize = True
follow = True

# debugging
debug_prints = False
iteration_print_frequency = 500

# saving
savefile_name=f"ae_{nsm.name}_k={k}_n={n_agents}_steps={n_steps}_dov={np.round(degrees_of_vision,2)}_r={radius}"
save_frequency=1 
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

sim = ae.SwarmSimulation(num_agents=n_agents, 
                         num_steps=n_steps, 
                         env_size=env_size, 
                         degrees_of_vision=degrees_of_vision,
                         radius=radius,
                         neighbour_selection_mechanism=nsm,
                         k=k,
                         visualize=visualize, 
                         follow=follow, 
                         graph_freq=graph_freq,
                         debug_prints=debug_prints,
                         iteration_print_frequency=iteration_print_frequency,
                         savefile_name=savefile_name,
                         save_frequency=save_frequency,
                         results_dir=results_dir)
sim.run()

print("Done")
