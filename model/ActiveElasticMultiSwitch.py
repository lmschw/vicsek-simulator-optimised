import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, circmean, circvar
import pickle
from datetime import datetime
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import services.ServiceAEHelper as saeh
import services.ServiceVicsekHelper as svh
import services.ServiceOrientations as sor
import services.ServiceSavedModel as ssm
import services.ServiceThresholdEvaluation as ste

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism as nsm
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod
from enums.EnumSwitchType import SwitchType

# AE Constants
EPSILON = 12
SIGMA = 0.7
SIGMA_MIN = 0.7
SIGMA_MAX = 0.7
UC = 0.05
UMAX = 0.1
WMAX = np.pi / 2
ALPHA = 2.0
BETA = 0.5
BETA_LEADER = 4
GAMMA = 1.0
G_GAIN = 1
K1 = 0.6
K2 = 0.05
L_0 = 0.5
K_REP = 2.0
SENSE_RANGE = 2.0
B_SENSE_RANGE = 0.5
DT = 0.05
DES_DIST = SIGMA * 2**(1/2)

SPACING = 0.8

class SwarmSimulation:
    def __init__(self, num_agents=7, num_steps=1000, env_size=25, degrees_of_vision=2*np.pi, radius=np.inf,
                 neighbour_selection_mechanism=nsm.NEAREST, k=np.inf, events=None, switch_summary=None,
                 activation_time_delays=[], is_activation_time_delay_relevant_for_events=False,  
                 colour_type=None, threshold_evaluation_method=ThresholdEvaluationMethod.LOCAL_ORDER,
                 update_if_no_neighbours=True, return_histories=False,
                 visualize=True, follow=True, graph_freq=5, debug_prints=False,
                 iteration_print_frequency=None, savefile_name=None, save_frequency=1, results_dir=None):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.env_size = env_size
        self.degrees_of_vision = degrees_of_vision
        self.radius = radius
        self.nsm = neighbour_selection_mechanism
        self.k = k
        self.events = events
        self.switch_summary = switch_summary
        self.activation_time_delays = activation_time_delays
        self.is_activation_time_delay_relevant_for_events = is_activation_time_delay_relevant_for_events
        self.colour_type = colour_type
        self.threshold_evaluation_method = threshold_evaluation_method
        self.update_if_no_neighbours = update_if_no_neighbours
        self.return_histories = return_histories
        self.visualize = visualize
        self.follow = follow
        self.graph_freq = graph_freq
        self.debug_prints = debug_prints
        self.iteration_print_frequency = iteration_print_frequency
        self.savefile_name = savefile_name
        self.save_frequency = save_frequency
        self.results_dir = results_dir
        self.save_path = f"{results_dir}/{savefile_name}"
        self.curr_agents = None
        self.centroid_trajectory = []
        self.states = []
        self.threshold_evaluation_choice_values_history = []
        self.initialize()

    def initialize(self):
        # Preparation of constants
        self.min_replacement_value = -1
        self.max_replacement_value = np.inf
        self.disorder_placeholder = -1
        self.order_placeholder = -2

        self.init_agents(self.num_agents)
        self.sigmas = np.full(self.num_agents, SIGMA)
        self.current_step = 0

        # Setup graph
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x - 7.5, centroid_x + 7.5)
            self.ax.set_ylim(centroid_y - 7.5, centroid_y + 7.5)
        else:
            self.ax.set_xlim(-self.env_size, self.env_size)
            self.ax.set_ylim(-self.env_size, self.env_size)

        # making sure we're not picking more neighbours than we could conceivable have
        if self.k == np.inf:
            self.k = self.num_agents

        # Preparation of active switchTypes
        if self.switch_summary:
            self.switch_types = [k for k, v in self.switch_summary.actives.items() if v == True]
        else:
            self.switch_types = []

        self.example_id = None

    def initialise_switching_values(self):
        """
        Initialises the valus that may be affected by switching: neighbour selection mechanisms, ks, speeds, time delays.

        Params:
            None

        Returns:
            Numpy arrays containing the neighbour selection mechanisms, ks and time delays for each particle.
        """

        nsms = np.full(self.num_agents, self.order_placeholder)
        nsmsDf = pd.DataFrame(nsms, columns=["nsms"])
        nsmsDf["nsms"] = nsmsDf["nsms"].replace(self.order_placeholder, self.nsm.value)
        nsms = np.array(nsmsDf["nsms"])

        ks = np.array(self.num_agents * [self.k])
        activation_time_delays = np.ones(self.num_agents)

        if self.switch_summary != None:
            info = self.switch_summary.getBySwitchType(SwitchType.NEIGHBOUR_SELECTION_MECHANISM)
            if info != None and info.initialValues != None:
                nsms = info.initialValues
            info = self.switch_summary.getBySwitchType(SwitchType.K)
            if info != None and info.initialValues != None:
                ks = info.initialValues
            info = self.switch_summary.getBySwitchType(SwitchType.ACTIVATION_TIME_DELAY)
            if info != None and info.initialValues != None:
                activation_time_delays = info.initialValues

        if len(self.activation_time_delays) > 0:
            activation_time_delays = self.activationTimeDelays

        return nsms, ks, activation_time_delays

    def init_agents(self, n_agents):
        rng = np.random
        n_points_x = n_agents
        n_points_y = n_agents
        
        init_x = 0
        init_y = 0

        x_values = np.linspace(init_x, init_x + (n_points_x - 1) * SPACING, n_points_x)
        y_values = np.linspace(init_y, init_y + (n_points_y - 1) * SPACING, n_points_y)
        xx, yy = np.meshgrid(x_values, y_values)

        pos_xs = xx.ravel() + (rng.random(n_points_x * n_points_y) * SPACING * 0.5) - SPACING * 0.25
        pos_ys = yy.ravel() + (rng.random(n_points_x * n_points_y) * SPACING * 0.5) - SPACING * 0.25
        pos_hs = (rng.random(n_points_x * n_points_x) * np.pi * 2) - np.pi

        num_agents = len(pos_xs)
        self.num_agents = num_agents

        nsms, ks, activation_time_delays = self.initialise_switching_values()

        self.curr_agents = np.column_stack([pos_xs, pos_ys, pos_hs])
        self.blocked = np.full(self.num_agents, False)
        self.nsms = nsms
        self.ks = ks
        self.activation_time_delays = activation_time_delays


    def graph_agents(self):
        """
        Visualizes the state of the simulation with matplotlib

        """  
        self.ax.clear()

        # Draw followers
        self.ax.scatter(self.curr_agents[:, 0], self.curr_agents[:, 1], color="white", s=15)
        self.ax.quiver(self.curr_agents[:, 0], self.curr_agents[:, 1],
                    np.cos(self.curr_agents[:, 2]), np.sin(self.curr_agents[:, 2]),
                    color="white", width=0.005, scale=40)

        # Draw Trajectory
        if len(self.centroid_trajectory) > 1:
            x_traj, y_traj = zip(*self.centroid_trajectory)
            self.ax.plot(x_traj, y_traj, color="orange")

        self.ax.set_facecolor((0, 0, 0))

        centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
        if self.follow:
            self.ax.set_xlim(centroid_x-10, centroid_x+10)
            self.ax.set_ylim(centroid_y-10, centroid_y+10)
        else:
            self.ax.set_xlim(-self.env_size, self.env_size)
            self.ax.set_ylim(-self.env_size, self.env_size)

        plt.pause(0.000001)

    def get_parameter_summary(self):
        """
        Creates a summary of all the model parameters ready for use for conversion to JSON or strings.

        Parameters:
            - asString (bool, default False) [optional]: if the summary should be returned as a dictionary or as a single string
        
        Returns:
            A dictionary or a single string containing all model parameters.
        """
        summary = {"n": self.num_agents,
                    "k": self.k,
                    "radius": self.radius,
                    "neighbourSelectionMechanism": self.nsm.name,
                    "domainSize": [self.env_size,self.env_size],
                    "tmax": self.num_steps,
                    "dt": DT,
                    "degreesOfVision": self.degrees_of_vision,
                    "activationTimeDelays": self.activation_time_delays,
                    "isActivationTimeDelayRelevantForEvents": self.is_activation_time_delay_relevant_for_events,
                    }
     
        if self.colour_type != None:
            summary["colourType"] = self.colour_type.value
            if self.example_id != None:
                summary["exampleId"] = self.example_id.tolist()

        if self.switch_summary != None:
            summary["switchSummary"] = self.switch_summary.getParameterSummary()
   
        if self.events:
            events_summary = []
            for event in self.events:
                events_summary.append(event.getParameterSummary())
            summary["events"] = events_summary
        return summary 

    def wrap_to_pi(self, x):
        """
        Wrapes the angles to [-pi, pi]

        """
        x = x % (np.pi * 2)
        x = (x + (np.pi * 2)) % (np.pi * 2)

        x[x > np.pi] = x[x > np.pi] - (np.pi * 2)

        return x

    def compute_distances_and_angles(self):
        """
        Computes and returns the distances and its x and y elements for all pairs of agents

        """
        headings = self.curr_agents[:, 2]

        # Build meshgrid 
        pos_xs = self.curr_agents[:, 0]
        pos_ys = self.curr_agents[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        # Calculate distances
        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > SENSE_RANGE] = np.inf
        #distances[distances == 0.0] = np.inf
        if self.debug_prints:
            print(f"Dists: {distances}")
        
        # Calculate angles in the local frame of reference
        headings = self.curr_agents[:, 2]
        angles = np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis]
        #if self.debug_prints:
        # print(f"Angles: {angles}")

        return distances, angles
    
    def __get_picked_neighbour_indices(self, sorted_indices, ks):
        """
        Chooses the indices of the neighbours that will be considered for updates.

        Params:
            - sorted_indices (arrays of ints): the sorted indices of all neighbours
            - ks (array of int): the current values of k for every individual

        Returns:
            Array containing the selected indices for each individual.
        """
        if self.switch_summary != None and self.switch_summary.isActive(SwitchType.K):
            k_switch = self.switch_summary.getBySwitchType(SwitchType.K)
            k_min, k_max = self.switch_summary.getMinMaxValuesForKSwitchIfPresent()
            
            candidates_order = sorted_indices[:, :k_switch.orderSwitchValue]
            if k_switch.orderSwitchValue < k_max:
                candidates_order = svh.padArray(candidates_order, self.num_agents, kMin=k_min, kMax=k_max)

            candidates_disorder = sorted_indices[:, :k_switch.disorderSwitchValue]
            if k_switch.disorderSwitchValue < k_max:
                candidates_disorder = svh.padArray(candidates_disorder, self.num_agents, kMin=k_min, kMax=k_max)

            candidates = np.where(((ks == k_switch.orderSwitchValue)[:, None]), candidates_order, candidates_disorder)
        elif len(ks) > 0:
            k_min, k_max = np.min(ks), np.max(ks)
            candidatesMax = sorted_indices[:, :k_max]
            candidatesMin = sorted_indices[:, :k_min]
            candidatesMin = svh.padArray(candidatesMin, self.num_agents, kMin=k_min, kMax=k_max)
            candidates = np.where(((ks == k_max)[:, None]), candidatesMax, candidatesMin)
        else:
            candidates = sorted_indices[:, :self.k]
        return candidates
    
    def __check_picked_for_neighbourhood(self, pos_diff, candidates, k_max_present):
        """
        Verifies that all the selected neighbours are within the perception radius.

        Params:
            - pos_diff (array of arrays of float): the position difference between every pair of individuals
            - candidates (array of int): the indices of the selected neighbours
            - k_max_present (int): waht is the highest value of k present in the current values of k

        Returns:
            An array of int indices of the selected neighbours that are actually within the neighbourhood.
        """
        if len(candidates) == 0 or len(candidates[0]) == 0:
            return candidates
        # exclude any individuals that are not neighbours
        picked_distances = np.take_along_axis(pos_diff, candidates, axis=1)
        minus_ones = np.full((self.num_agents,k_max_present), -1)
        picked = np.where(((candidates == -1) | (picked_distances > self.radius**2)), minus_ones, candidates)
        return picked
    
    def __create_boolean_mask_from_picked_neighbour_indices(self, picked, k_max):
        """
        Creates a boolean mask from the indices of the selected neighbours.

        Params:
            - picked (array of array of int): the selected indices for each individual

        Returns:
            An array of arrays of booleans representing which neighbours have been selected by each individual.
        """
        if len(picked) == 0 or len(picked[0]) == 0:
            return np.full((self.num_agents, self.num_agents), False)
        # create the boolean mask
        ns = np.full((self.num_agents,self.num_agents+1), False) # add extra dimension to catch indices that are not applicable
        picked_values = np.full((self.num_agents, k_max), True)
        np.put_along_axis(ns, picked, picked_values, axis=1)
        ns = ns[:, :-1] # remove extra dimension to catch indices that are not applicable
        return ns
    
    def __get_picked_neighbours(self, pos_diff, candidates, ks, is_min):
        """
        Determines which neighbours the individuals should consider.

        Params:
            - posDiff (array of arrays of floats): the distance from every individual to all other individuals
            - candidates (array of arrays of floats): represents either the position distance between each pair of individuals or a fillValue if they are not neighbours  
            - ks (array of ints): which value of k every individual observes
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        
        k_max = np.max(ks)

        if self.switch_summary != None and self.switch_summary.isActive(SwitchType.K):
            _, k_max = self.switch_summary.getMinMaxValuesForKSwitchIfPresent()

        sorted_indices = candidates.argsort(axis=1)
        if is_min == False:
            sorted_indices = np.flip(sorted_indices, axis=1)
        
        picked = self.__get_picked_neighbour_indices(sorted_indices=sorted_indices,  ks=ks)
        picked = self.__check_picked_for_neighbourhood(pos_diff=pos_diff, candidates=picked, k_max_present=k_max)
        mask = self.__create_boolean_mask_from_picked_neighbour_indices(picked, k_max)
        return mask        
            
    def pick_position_neighbours(self, positions, neighbours, ks, is_min=True):
        """
        Determines which neighbours the individuals should considered based on the neighbour selection mechanism and k with regard to position.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every 
            - ks (array of ints): which value of k every individual observes
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        pos_diff = saeh.getPositionDifferences(positions)
        if is_min == True:
            fill_value = self.max_replacement_value
        else:
            fill_value = self.min_replacement_value

        fill_vals = np.full((self.num_agents,self.num_agents), fill_value)
        candidates = np.where((neighbours), pos_diff, fill_vals)

        # select the best candidates
        return self.__get_picked_neighbours(pos_diff=pos_diff, candidates=candidates, ks=ks, is_min=is_min)
    
    def pick_orientation_neighbours(self, positions, orientations, neighbours, ks, is_min=True):
        """
        Determines which neighbours the individuals should consider based on the neighbour selection mechanism and k with regard to orientation.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes
            - is_min (boolean) [optional, default=True]: whether to take the least orientionally different or most orientationally different neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        pos_diff = saeh.getPositionDifferences(positions)
        orient_diff = saeh.getOrientationDifferences(orientations)

        if is_min == True:
            fill_value = self.max_replacement_value
        else:
            fill_value = self.min_replacement_value

        fill_vals = np.full((self.num_agents,self.num_agents), fill_value)
        candidates = np.where((neighbours), orient_diff, fill_vals)

        # select the best candidates
        return self.__get_picked_neighbours(pos_diff=pos_diff, candidates=candidates, ks=ks, isMin=is_min)
    
    def pick_random_neighbours(self, positions, neighbours, ks):
        """
        Determines which neighbours the individuals should consider based on random selection.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        np.fill_diagonal(neighbours, False)
        pos_diff = saeh.getPositionDifferences(positions)
        k_max = np.max(ks)
        
        candidate_indices = svh.getIndicesForTrueValues(neighbours, paddingType='repetition')
        rng = np.random.default_rng()
        rng.shuffle(candidate_indices, axis=1)
        if self.switch_summary != None and self.switch_summary.isActive(SwitchType.K):
            k_min, k_max = self.switch_summary.getMinMaxValuesForKSwitchIfPresent()
            if len(candidate_indices[0]) < k_max:
                candidate_indices = svh.padArray(candidate_indices, self.num_agents, k_min, k_max)
        elif k_max < self.k:
            candidate_indices = svh.padArray(candidate_indices, self.num_agents, k_max, self.k)
        elif len(candidate_indices[0]) < k_max:
            candidate_indices = svh.padArray(candidate_indices, self.num_agents, len(candidate_indices[0]), k_max)
        picked = self.__get_picked_neighbour_indices(sorted_indices=candidate_indices, ks=ks)
        picked = self.__check_picked_for_neighbourhood(pos_diff=pos_diff, candidates=picked, k_max_present=k_max)
        selection = self.__create_boolean_mask_from_picked_neighbour_indices(picked, k_max)
        np.fill_diagonal(selection, True)
        return selection
    
    def get_nsm_neighbours_mask(self, neighbour_selection_mechanism, orientations, neighbours, ks):
        positions = self.curr_agents[:,:2]
        match neighbour_selection_mechanism:
            case nsm.NEAREST:
                pickedNeighbours = self.pick_position_neighbours(positions, neighbours, ks, is_min=True)
            case nsm.FARTHEST:
                pickedNeighbours = self.pick_position_neighbours(positions, neighbours, ks, is_min=False)
            case nsm.LEAST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pick_orientation_neighbours(positions, orientations, neighbours, ks, is_min=True)
            case nsm.HIGHEST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pick_orientation_neighbours(positions, orientations, neighbours, ks, is_min=False)
            case nsm.RANDOM:
                pickedNeighbours = self.pick_random_neighbours(positions, neighbours, ks)
            case nsm.ALL:
                pickedNeighbours = neighbours
        return pickedNeighbours
    
    def get_decisions(self, t, neighbours, threshold_evaluation_choice_values, previous_threshold_evaluation_choice_values, switch_type, switch_type_values, blocked):
        """
        Computes whether the individual chooses to use option A or option B as its value based on the local order, 
        the average previous local order and a threshold.

        Params:
            - t (int): the current timestep
            - thresholdEvaluationChoiceValues (array of floats): the local order from the point of view of every individual
            - previousthresholdEvaluationChoiceValues of arrays of floats): the local order for every individual at every previous time step
            - switchType (SwitchType): the property that the values are assigned to
            - switchTypeValues (array of ints): the current switchTypeValue selection for every individual
            - blocked (array of booleans): whether updates are possible

        Returns:
            Numpy array containing the updated switchTypeValues for every individual for the given switchType.
        """
        switch_info = self.switch_summary.getBySwitchType(switch_type)
        switch_difference_threshold_lower = switch_info.lowerThreshold
        switch_difference_threshold_upper = switch_info.upperThreshold

        prev = np.average(previous_threshold_evaluation_choice_values[max(t-switch_info.numberPreviousStepsForThreshold, 0):t+1], axis=0)

        old_with_new_order_values = np.where(((threshold_evaluation_choice_values >= switch_difference_threshold_upper) & (prev <= switch_difference_threshold_upper) & (blocked != True)), np.full(len(switch_type_values), switch_info.getOrderValue()), switch_type_values)
        updated_switch_values = np.where(((threshold_evaluation_choice_values <= switch_difference_threshold_lower) & (prev >= switch_difference_threshold_lower) & (blocked != True)), np.full(len(switch_type_values), switch_info.getDisorderValue()), old_with_new_order_values)
        if self.update_if_no_neighbours == False:
            neighbour_counts = np.count_nonzero(neighbours, axis=1)
            updated_switch_values = np.where((neighbour_counts <= 1), switch_type_values, updated_switch_values)
        return updated_switch_values
    
    def prepare_ks(self, ks):
        if self.switch_summary != None and self.switch_summary.isActive(SwitchType.K):
            ks = ks
        else:
            ks = np.array(self.num_agents * [self.k])
        return ks
    
    def get_neighbour_mask(self, neighbour_candidate_mask, orientations):
        ks = self.prepare_ks(self.ks)

        if self.switch_summary != None and self.switch_summary.isActive(SwitchType.NEIGHBOUR_SELECTION_MECHANISM):
            nsms_switch = self.switch_summary.getBySwitchType(SwitchType.NEIGHBOUR_SELECTION_MECHANISM)
            neighbours_order = self.get_nsm_neighbours_mask(neighbour_selection_mechanism=nsms_switch.orderSwitchValue,
                                                               orientations=orientations,
                                                               neighbours=neighbour_candidate_mask,
                                                               ks=ks)
            neighbours_disorder = self.get_nsm_neighbours_mask(neighbour_selection_mechanism=nsms_switch.disorderSwitchValue,
                                                                                    orientations=orientations,
                                                                                    neighbours=neighbour_candidate_mask,
                                                                                    ks=ks)
            picked_neighbours = np.where(((self.nsms == nsms_switch.orderSwitchValue.value)), neighbours_order, neighbours_disorder)
        else:
            picked_neighbours = self.get_nsm_neighbours_mask(neighbour_selection_mechanism=self.nsm,
                                                                                      orientations=orientations, 
                                                                                      neighbours=neighbour_candidate_mask,
                                                                                      ks=ks)

        np.fill_diagonal(picked_neighbours, True)
        return picked_neighbours
    
    def get_neighbours(self):
        positions = self.curr_agents[:,:2]
        orientations = sor.computeUvCoordinatesForList(self.curr_agents[:,2])
        is_neighbours = saeh.getNeighboursWithLimitedVision(positions=positions, orientations=orientations,
                                                                radius=self.radius, degreesOfVision=self.degrees_of_vision)
        
        # update the choice of switch value before creating the indices for the selected neighbours
        if self.switch_summary != None:
            threshold_evaluation_choice_values = ste.getThresholdEvaluationValuesForChoice(thresholdEvaluationMethod=self.threshold_evaluation_method, 
                                                                                           positions=positions, 
                                                                                           orientations=orientations, 
                                                                                           neighbours=is_neighbours, 
                                                                                           domainSize=self.env_size)

            if self.current_step > 1 and self.current_step < 5:
                print(f"{self.current_step}: {threshold_evaluation_choice_values}")
            self.threshold_evaluation_choice_values_history.append(threshold_evaluation_choice_values)
        
            if SwitchType.NEIGHBOUR_SELECTION_MECHANISM in self.switch_summary.switches.keys():
                self.nsms = self.get_decisions(t=self.current_step, 
                                                           neighbours=is_neighbours, 
                                                           threshold_evaluation_choice_values=threshold_evaluation_choice_values,
                                                           previous_threshold_evaluation_choice_values=self.threshold_evaluation_choice_values_history, 
                                                           switch_type=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
                                                           switch_type_values=self.nsms, 
                                                           blocked=self.blocked)
            if SwitchType.K in self.switch_summary.switches.keys():
                self.ks = self.get_decisions(t=self.current_step, 
                                                           neighbours=is_neighbours, 
                                                           threshold_evaluation_choice_values=threshold_evaluation_choice_values, 
                                                           previous_threshold_evaluation_choice_values=self.threshold_evaluation_choice_values_history, 
                                                           switch_type=SwitchType.K, 
                                                           switch_type_values=self.ks, 
                                                           blocked=self.blocked)
            if SwitchType.ACTIVATION_TIME_DELAY in self.switch_summary.switches.keys():
                self.activation_time_delays = self.get_decisions(t=self.current_step, 
                                                           neighbours=is_neighbours, 
                                                           threshold_evaluation_choice_values=threshold_evaluation_choice_values, 
                                                           previous_threshold_evaluation_choice_values=self.threshold_evaluation_choice_values_history, 
                                                           switch_type=SwitchType.ACTIVATION_TIME_DELAY, 
                                                           switch_type_values=self.activation_time_delays, 
                                                           blocked=self.blocked)

        # TODO get the indices according to the ks and nsms in the curr_agents instead of only according to the general nsm
        mask = self.get_neighbour_mask(is_neighbours, orientations)

        return mask * is_neighbours
    
    def create_boolean_mask(self, picked, kMax):
        if len(picked) == 0 or len(picked[0]) == 0:
            return np.full((self.num_agents, self.num_agents), False)
        # create the boolean mask
        ns = np.full((self.num_agents,self.num_agents+1), False) # add extra dimension to catch indices that are not applicable
        pickedValues = np.full((self.num_agents, kMax), True)
        np.put_along_axis(ns, picked, pickedValues, axis=1)
        ns = ns[:, :-1] # remove extra dimension to catch indices that are not applicable
        return ns
    
    # TODO: fix event handling for AE
    def handle_events(self, t, nsms, ks, speeds, activation_time_delays):
        """
        Handles all types of events.

        Params:
            - t (int): the current timestep
            - nsms (array of NeighbourSelectionMechanism): how every particle selects its neighbours at the current timestep
            - ks (array of ints): how many neighbours each particle considers at the current timestep
            - speeds (array of floats): how fast each particle moves at the current timestep
            - activationTimeDelays (array of ints): how often a particle is ready to update its orientation

        Returns:
            Arrays containing the updates orientations, neighbour selecton mechanisms, ks, speeds, which particles are blocked from updating and the colours assigned to each particle.
        """
        positions = self.curr_agents[:,:2]
        orientations = sor.computeUvCoordinatesForList(self.curr_agents[:,2])
        blocked = np.full(self.num_agents, False)
        colours = np.full(self.num_agents, 'k')
        if self.events != None:
                for event in self.events:
                    orientations, nsms, ks, speeds, blocked, colours = event.check(self.num_agents, t, positions, orientations, nsms, ks, speeds, DT, activation_time_delays, self.isActivationTimeDelayRelevantForEvents, self.colourType)
        return orientations, nsms, ks, speeds, blocked, colours

    def get_pi_elements(self, distances, angles, neighbour_mask):
        """
        Calculates the x and y components of the proximal control vector

        """  
        distances[distances == np.inf] = 0.0
        dists = neighbour_mask * distances
        dists[dists == 0.0] = np.inf
        forces = -EPSILON * (2 * (self.sigmas[:, np.newaxis] ** 4 / dists ** 5) - (self.sigmas[:, np.newaxis] ** 2 / dists ** 3))
        forces[dists == np.inf] = 0.0

        #if self.debug_prints:
        # print(f"Forces: {forces}")

        p_x = np.sum(np.multiply(forces, np.cos(angles)), axis=1)
        p_y = np.sum(np.multiply(forces, np.sin(angles)), axis=1)

        return p_x, p_y
    
    def get_hi_elements(self, neighbour_mask):
        """
        Calculates the x and y components of the heading alignment vector
        using a per-agent neighbour mask.
        """
        headings = self.curr_agents[:, 2]

        # Precompute trig functions
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)

        # Masked sums per agent
        alignment_coss = np.sum(neighbour_mask * cos_h[None, :], axis=1)
        alignment_sins = np.sum(neighbour_mask * sin_h[None, :], axis=1)

        alignment_angs = np.arctan2(alignment_sins, alignment_coss)
        alignment_mags = np.sqrt(alignment_coss**2 + alignment_sins**2)

        h_x = alignment_mags * np.cos(alignment_angs - headings)
        h_y = alignment_mags * np.sin(alignment_angs - headings)

        return h_x, h_y

    def compute_fi(self):
        """
        Computes the virtual force vector components

        """  
        dists, angles = self.compute_distances_and_angles()

        neighbours = self.get_neighbours()

        p_x, p_y = self.get_pi_elements(dists, angles, neighbours)
        h_x, h_y = self.get_hi_elements(neighbours)

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 

        #if self.debug_prints:
        # print(f"Fx: {f_x}")
        # print(f"Fy: {f_y}")

        return f_x, f_y
    
    def compute_u_w(self, f_x, f_y):
        """
        Computes u and w given the components of Fi

        """
        u = K1 * f_x + UC
        u[u > UMAX] = UMAX
        u[u < 0] = 0.0

        w = K2 * f_y
        w[w > WMAX] = WMAX
        w[w < -WMAX] = -WMAX

        return u, w
    
    def update_agents(self):
        """
        Updates agents

        """  
        # Calculate forces
        f_x, f_y = self.compute_fi()
        u, w = self.compute_u_w(f_x, f_y)

        # Project to local frame
        headings = self.curr_agents[:, 2]
        x_vel = np.multiply(u, np.cos(headings))
        y_vel = np.multiply(u, np.sin(headings))
        # print(f"X add: {x_vel}")
        # print(f"Y add: {y_vel}")

        # Update agents
        self.curr_agents[:, 0] = self.curr_agents[:, 0] + x_vel * DT
        self.curr_agents[:, 1] = self.curr_agents[:, 1] + y_vel * DT
        self.curr_agents[:, 2] = self.wrap_to_pi(self.curr_agents[:, 2] + w * DT)

    def run(self):

        if self.savefile_name and self.save_frequency:
            ssm.logModelParams(path=f"{self.results_dir}/{self.savefile_name}_modelParams", modelParamsDict=self.get_parameter_summary())
            ssm.initialiseCsvFileHeaders(path=f"{self.results_dir}/{self.savefile_name}")

        while self.current_step < self.num_steps / DT:

            # Update simulation
            self.update_agents()

            # Update experiment data
            if self.return_histories:
                self.states.append(self.curr_agents.copy())
            centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            if self.iteration_print_frequency and not (self.current_step % self.iteration_print_frequency):
                print(f"------------------------ Iteration {self.current_step} ------------------------")

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents()

            if self.savefile_name and self.save_frequency and self.current_step % self.save_frequency == 0:
                switchValues = {'nsms': self.nsms, 'ks': self.ks, 'activationTimeDelays': self.activation_time_delays}
                ssm.saveModelTimestep(timestep=self.current_step, 
                                        positions=self.curr_agents[:,:2], 
                                        orientations=sor.computeUvCoordinatesForList(self.curr_agents[:,2]),
                                        colours=np.full(self.num_agents, 'b'),
                                        path=self.save_path,
                                        switchValues=switchValues,
                                        switchTypes=self.switch_types)

            self.current_step +=1
        return self.states