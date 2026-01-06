import numpy as np

import model.ActiveElastic as ae
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

# environment and simulation parameters
n_agents = 3
n_steps = 10000
env_size = 100

# perception parameters
degrees_of_vision = 2*np.pi
radius = np.inf

# neighbour selection parameters
nsm = NeighbourSelectionMechanism.ALL
k = 9

# visualisation parameters
graph_freq = 10
visualize = True
follow = True

sim = ae.SwarmSimulation(num_agents=n_agents, 
                         num_steps=n_steps, 
                         env_size=env_size, 
                         degrees_of_vision=degrees_of_vision,
                         radius=radius,
                         neighbour_selection_mechanism=nsm,
                         k=k,
                         visualize=visualize, 
                         follow=follow, 
                         graph_freq=graph_freq)
sim.run()