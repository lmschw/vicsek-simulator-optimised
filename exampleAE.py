import numpy as np
import os

import model.ActiveElastic as ae
import services.ServiceSavedModel as ssm
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

# environment and simulation parameters
n_agents = 3
n_steps = 10000
env_size = 100

# perception parameters
degrees_of_vision = 2*np.pi
radius = np.inf

# neighbour selection parameters
nsm = NeighbourSelectionMechanism.NEAREST
k = 5

# visualisation parameters
graph_freq = 10
visualize = False
follow = False

# debugging
debug_prints = False
iteration_print_frequency = 1000

# saving
save_frequency=1 
results_dir = os.path.join(os.path.dirname(__file__), "results", "ae")
os.makedirs(results_dir, exist_ok=True)

num_iters = 50

for nsm in [NeighbourSelectionMechanism.NEAREST,
            NeighbourSelectionMechanism.FARTHEST,
            NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
            NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]:
    for k in [1,2,3,4,5]:
        for iter in range(1,num_iters+1):
            savefile_name=f"ae_{nsm.name}_k={k}_n={n_agents}_steps={n_steps}_dov={np.round(degrees_of_vision,2)}_r={radius}_{iter}"

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
            print(f"Done with nsm={nsm.name}, k={k} iteration {iter}/{num_iters}")
        print(f"Done with nsm={nsm.name}, k={k}")
    print(f"Done with nsm={nsm.name}")
print("Done")
