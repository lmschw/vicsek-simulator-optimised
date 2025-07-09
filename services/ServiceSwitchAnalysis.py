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
                        switched_after_by_id[i].append(15000)
                    else:
                        switched_after_by_id[i] = [15000]  # If not exposed, set to a large value (e.g., 15000)
    switched_after_by_time = {}
    for k in switched_after_by_id.keys():
        for v in switched_after_by_id[k]:
            if v in switched_after_by_time:
                switched_after_by_time[v] += 1
            else:
                switched_after_by_time[v] = 1
    return switched_after_by_id, switched_after_by_time

def compute_infection_percentage_grid_values(simulation_data, switch_values, target_switch_value, domain_size, num_cells=None, start_t=None, end_t=None):
    times, positions, _ = simulation_data
    #grid, x, y = create_grid(domain_size=domain_size, num_cells=num_cells)
    if start_t == None:
        start_t = 0
    if end_t == None:
        end_t = len(positions)
    if num_cells == None:
        num_cells = int(domain_size[0]**0.5) * int(domain_size[1]**0.5)
    domain_area = domain_size[0] * domain_size[1]
    pointArea = domain_area / num_cells
    length = np.sqrt(pointArea)
    max_x = int(domain_size[0]/length)
    max_y = int(domain_size[1]/length)
    grid_values = np.zeros((end_t-start_t, max_x, max_y, 1))

    for t in range(start_t, end_t):
        for x in range(max_x):
            for y in range(max_y):
                candSwitch = [switch_values[t][part] == target_switch_value for part in range(len(positions[t])) if positions[t][part][0] <= ((x+1) * length) and positions[t][part][0] >= (x * length) and positions[t][part][1] <= ((y+1) * length) and positions[t][part][1] >= (y * length)]
                grid_values[t][x][y] = np.count_nonzero(candSwitch)/len(candSwitch) if len(candSwitch) > 0 else 0
    return times, grid_values


def get_affected_individuals(positions, event_origin_point, event_radius, domain_size):
    posWithCenter = np.zeros((len(positions)+1, 2))
    posWithCenter[:-1] = positions
    posWithCenter[-1] = event_origin_point
    rij2 = ServiceVicsekHelper.getDifferences(posWithCenter, domain_size)
    relevantDistances = rij2[-1][:-1] # only the comps to the origin and without the origin point
    candidates = (relevantDistances <= event_radius**2)
    return np.argwhere(candidates == True).flatten()

def create_grid(domain_size, num_cells=None):
    if num_cells == None:
        num_cells = int(domain_size[0]**0.5) * int(domain_size[1]**0.5)
    domain_area = domain_size[0] * domain_size[1]
    pointArea = domain_area / num_cells
    length = np.sqrt(pointArea)

    cells = []
    for x in np.arange(0, domain_size[0], length):
        for y in np.arange(0, domain_size[1], length):
            cells.append([(x, y), (x+length, y+length)])
    return cells, int(domain_size[0]/length), int(domain_size[1]/length)