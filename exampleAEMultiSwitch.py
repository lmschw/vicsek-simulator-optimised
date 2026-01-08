import numpy as np
import os

import model.ActiveElasticMultiSwitch as ae
from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary
import services.ServiceSavedModel as ssm
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod
from enums.EnumSwitchType import SwitchType

# environment and simulation parameters
n_agents = 3
n_steps = 10000
env_size = 25

# perception parameters
degrees_of_vision = 2*np.pi
radius = np.inf

# neighbour selection parameters
nsm = NeighbourSelectionMechanism.NEAREST
k = 5
update_if_no_neighbours = True
threshold_evaluation_method = ThresholdEvaluationMethod.LOCAL_ORDER
is_activation_time_delay_relevant_for_events = False
activation_time_delays = []

# Switch Summary
nsm_switch = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM,
                                    values=[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.NEAREST], 
                                    thresholds=[0.1], 
                                    numberPreviousStepsForThreshold=100, 
                                    initialValues=None)
switch_summary = SwitchSummary([nsm_switch])

# events
events = []

# visualisation parameters
graph_freq = 10
visualize = True
follow = False
colour_type = None

# debugging
debug_prints = False
iteration_print_frequency = 1000
return_histories = True

# saving
save_frequency=1 
results_dir = os.path.join(os.path.dirname(__file__), "results", "ae_switch")
os.makedirs(results_dir, exist_ok=True)

num_iters = 30

for iter in range(1,num_iters+1):
    savefile_name=f"ae_{nsm.name}_k={k}_n={n_agents}_steps={n_steps}_dov={np.round(degrees_of_vision,2)}_r={radius}_{iter}"

    sim = ae.SwarmSimulation(num_agents=n_agents, 
                            num_steps=n_steps, 
                            env_size=env_size, 
                            degrees_of_vision=degrees_of_vision,
                            radius=radius,
                            neighbour_selection_mechanism=nsm,
                            k=k,
                            events=events,
                            switch_summary=switch_summary,
                            activation_time_delays=activation_time_delays,
                            is_activation_time_delay_relevant_for_events=is_activation_time_delay_relevant_for_events,
                            colour_type=colour_type,
                            threshold_evaluation_method=threshold_evaluation_method,
                            update_if_no_neighbours=update_if_no_neighbours,
                            return_histories=return_histories,
                            visualize=visualize, 
                            follow=follow, 
                            graph_freq=graph_freq,
                            debug_prints=debug_prints,
                            iteration_print_frequency=iteration_print_frequency,
                            savefile_name=savefile_name,
                            save_frequency=save_frequency,
                            results_dir=results_dir)
    sim.run()
    print(f"Done with iteration {iter}/{num_iters}")
print("Done")
