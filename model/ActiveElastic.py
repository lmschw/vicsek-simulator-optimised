import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, circmean, circvar
import pickle
from datetime import datetime
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

import services.ServiceAEHelper as saeh
import services.ServiceOrientations as sor

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism as nsm


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

class SwarmSimulation:
    def __init__(self, num_agents=7, num_steps=1000, env_size=25, degrees_of_vision=2*np.pi, radius=np.inf,
                 neighbour_selection_mechanism=nsm.NEAREST, k=np.inf,
                 visualize=True, follow=True, graph_freq=5, debug_prints=False,
                 iteration_print_frequency=None):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.env_size = env_size
        self.degrees_of_vision = degrees_of_vision
        self.radius = radius
        self.nsm = neighbour_selection_mechanism
        self.k = k
        self.visualize = visualize
        self.follow = follow
        self.graph_freq = graph_freq
        self.debug_prints = debug_prints
        self.iteration_print_frequency = iteration_print_frequency
        self.curr_agents = None
        self.centroid_trajectory = []
        self.states = []
        self.initialize()

    def initialize(self):
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


    def init_agents(self, n_agents):
        rng = np.random
        n_points_x = n_agents
        n_points_y = n_agents
        spacing = 0.8
        init_x = 0
        init_y = 0

        x_values = np.linspace(init_x, init_x + (n_points_x - 1) * spacing, n_points_x)
        y_values = np.linspace(init_y, init_y + (n_points_y - 1) * spacing, n_points_y)
        xx, yy = np.meshgrid(x_values, y_values)

        pos_xs = xx.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_ys = yy.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_hs = (rng.random(n_points_x * n_points_x) * 3.1415926 * 2) - 3.1415926

        # pos_xs = np.array([-3, -2, -2, -2.1])
        # pos_ys = np.array([-2.5, -2, -3, -2.4])
        # pos_hs = np.array([0.5, 0.5, 0.5, 0.5])
        num_agents = len(pos_xs)
        self.num_agents = num_agents

        # making sure we're not picking more neighbours than we could conceivable have
        if self.k == np.inf:
            self.k = self.num_agents

        self.curr_agents = np.column_stack([pos_xs, pos_ys, pos_hs])

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
        # print(f"Angles: {angles}")

        return distances, angles
    
    def get_neighbours(self, distances):
        positions = self.curr_agents[:,:2]
        orientations = sor.computeUvCoordinatesForList(self.curr_agents[:,2])
        is_neighbours = saeh.getNeighboursWithLimitedVision(positions=positions, orientations=orientations,
                                                                radius=self.radius, degreesOfVision=self.degrees_of_vision)

        orientation_diffs = saeh.getOrientationDifferences(orientations)
        match self.nsm:
            case nsm.NEAREST:
                indices = np.argsort(distances)[:,:self.k]
            case nsm.FARTHEST:
                indices = np.argsort(-distances)[:,:self.k]
            case nsm.LEAST_ORIENTATION_DIFFERENCE:
                indices = np.argsort(orientation_diffs)[:,:self.k]
            case nsm.HIGHEST_ORIENTATION_DIFFERENCE:
                indices = np.argsort(-orientation_diffs)[:,:self.k]
            case nsm.ALL:
                indices = np.argsort(distances)
            case nsm.RANDOM:
                indices = np.argsort(distances)
                rng = np.random.default_rng()
                rng.shuffle(indices, axis=1)
                indices = indices[:,:self.k]

        mask = self.create_boolean_mask(indices, self.k)
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

    def get_pi_elements(self, distances, angles, neighbour_mask):
        """
        Calculates the x and y components of the proximal control vector

        """  
        distances[distances == np.inf] = 0.0
        dists = neighbour_mask * distances
        dists[dists == 0.0] = np.inf
        forces = -EPSILON * (2 * (self.sigmas[:, np.newaxis] ** 4 / dists ** 5) - (self.sigmas[:, np.newaxis] ** 2 / dists ** 3))
        forces[dists == np.inf] = 0.0

        """
        forces = -EPSILON * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances ** 3))
        forces[distances == np.inf] = 0.0
        neighbours = self.get_neighbours(distances)
        forces[neighbours == False] = 0.0
        """
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
    
    def get_gi_elements(self, distances, angles):
        """
        Calculates the x and y components of the goal direction vector

        """  
        pass

    def compute_fi(self):
        """
        Computes the virtual force vector components

        """  
        dists, angles = self.compute_distances_and_angles()

        neighbours = self.get_neighbours(dists)

        p_x, p_y = self.get_pi_elements(dists, angles, neighbours)
        h_x, h_y = self.get_hi_elements(neighbours)

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 
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
        Updates agents duhh

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
        while self.current_step < self.num_steps / DT:

            # Update simulation
            self.update_agents()
            self.current_step +=1

            # Update experiment data
            self.states.append(self.curr_agents.copy())
            centroid_x, centroid_y = np.mean(self.curr_agents[:, 0]), np.mean(self.curr_agents[:, 1])
            self.centroid_trajectory.append((centroid_x, centroid_y))

            if self.iteration_print_frequency and not (self.current_step % self.iteration_print_frequency):
                print(f"------------------------ Iteration {self.current_step} ------------------------")

            if not (self.current_step % self.graph_freq) and self.visualize and self.current_step > 0:
                self.graph_agents()



    # --------------------------------------------------------------------- Utils ---------------------------------------------------------------------

    def wrap_to_pi(self, x):
        """
        Wrapes the angles to [-pi, pi]

        """
        x = x % (3.1415926 * 2)
        x = (x + (3.1415926 * 2)) % (3.1415926 * 2)

        x[x > 3.1415926] = x[x > 3.1415926] - (3.1415926 * 2)

        return x
        

if __name__ == "__main__":
    n_agents = 3
    n_steps = 10000
    env_size = 100
    graph_freq = 10
    visualize = True
    follow = True
    
    sim = SwarmSimulation(n_agents, n_steps, env_size, visualize, follow, graph_freq)
    sim.run()


