import numpy as np
import services.ServiceVicsekHelper as ServiceVicsekHelper

def compute_time_between_exposure_and_switch(positions, switch_values, target_switch_value, event_start, event_origin_point, event_radius, domain_size):
    exposed_at = {}
    switched_after_by_id = {}
    for t in range(event_start, len(positions)):
        affected = get_affected_individuals(positions=positions[t], event_origin_point=event_origin_point, event_radius=event_radius, domain_size=domain_size)
        for i in affected:
            exposed_at[i] = t
        if t > 0:
            switched = np.argwhere((switch_values[t-1] != np.full(len(switch_values[t-1]), target_switch_value)) & (switch_values[t] == np.full(len(switch_values[t]), target_switch_value))).flatten()
            for i in switched:
                if i in exposed_at:
                    if i in switched_after_by_id:
                        switched_after_by_id[i].append(t-exposed_at[i])
                    else:
                        switched_after_by_id[i] = [t-exposed_at[i]]
                else:
                    if i in switched_after_by_id:
                        switched_after_by_id[i].append(-1)
                    else:
                        switched_after_by_id[i] = [-1]
    switched_after_by_time = {}
    for k in switched_after_by_id.keys():
        for v in switched_after_by_id[k]:
            if v in switched_after_by_time:
                switched_after_by_time[v] += 1
            else:
                switched_after_by_time[v] = 1
    return switched_after_by_id, switched_after_by_time


def get_affected_individuals(positions, event_origin_point, event_radius, domain_size):
    posWithCenter = np.zeros((len(positions)+1, 2))
    posWithCenter[:-1] = positions
    posWithCenter[-1] = event_origin_point
    rij2 = ServiceVicsekHelper.getDifferences(posWithCenter, domain_size)
    relevantDistances = rij2[-1][:-1] # only the comps to the origin and without the origin point
    candidates = (relevantDistances <= event_radius**2)
    return np.argwhere(candidates == True).flatten()