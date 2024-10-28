import time
import numpy as np
import pandas as pd
import math
import json, codecs
from sklearn.cluster import AgglomerativeClustering
from enum import Enum
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary
from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceMetric as ServiceMetric

radius = 20
density = 0.01
numIterationsPerRun = 5
tmax = 5000

# Parameters for the genetic algorithm
population_size = 10
# threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder
lower_bounds = [0, 0, 0, -1, -1, -1, -1]
upper_bounds = [0.5, tmax, 5, 5, 5, 5, 5]
generations = 20
mutation_rate = 1


# Evaluation time windows
orderTimeWindowsInitialOrder = [[0, 1000]]
disorderTimeWindowsInitialOrder = [[2001, tmax]]
orderTimeWindowsInitialDisorder = [[2001, tmax]]
disorderTimeWindowsInitialDisorder = [[0, 1000]]


nsmMapping = {0: NeighbourSelectionMechanism.ALL,
              1: NeighbourSelectionMechanism.NEAREST,
              2: NeighbourSelectionMechanism.FARTHEST,
              3: NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
              4: NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE,
              5: NeighbourSelectionMechanism.RANDOM}

def getNeighbourSelectionMechanismForId(nsmId):
    if nsmId == -1:
        return None
    return nsmMapping[nsmId]

def getThresholdEvaluationMethod(thresholdEvaluationMethodId):
    return list(ThresholdEvaluationMethod)[thresholdEvaluationMethodId]

def getSwitchInformation(switchType, threshold, previousSteps, disorderValue, orderValue):
    return SwitchInformation(
                            switchType=switchType, 
                            values=(orderValue, disorderValue),
                            thresholds=[threshold],
                            numberPreviousStepsForThreshold=previousSteps
                            )
        
def getSwitchSummary(threshold, previousSteps, nsmDisorder, nsmOrder, kDisorder, kOrder):
    switches = []
    if nsmDisorder != None and nsmOrder != None:
        switches.append(getSwitchInformation(SwitchType.NEIGHBOUR_SELECTION_MECHANISM, threshold, previousSteps, nsmDisorder, nsmOrder))
    if kDisorder != -1 and kOrder != -1:
        switches.append(getSwitchInformation(SwitchType.K, threshold, previousSteps, kDisorder, kOrder))
    return SwitchSummary(switches)
    

def getEvent(domainSize, radius, eventEffect):
    return ExternalStimulusOrientationChangeEvent(
                                                startTimestep=1000,
                                                duration=1000,  
                                                domainSize=domainSize, 
                                                eventEffect=eventEffect, 
                                                distributionType=DistributionType.LOCAL_SINGLE_SITE, 
                                                areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                                angle=np.pi,
                                                radius=radius,
                                                eventSelectionType=EventSelectionType.RANDOM
                                                )

def getEvents(domainSize, radius, eventEffect):
    return [getEvent(domainSize, radius, eventEffect)]

def getFinalOrderForSwitching(eventEffect, nsm, k, threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder, orderTimeWindows, disorderTimeWindows):
    domainSize = (50, 50)
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density, domainSize)
    noisePercentage = 1
    noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
    speed = 1
    degreesOfVision = 2*np.pi
    numberOfPreviousSteps = int(previousSteps)

    nsm = getNeighbourSelectionMechanismForId(nsm)
    nsmDisorder = getNeighbourSelectionMechanismForId(nsmDisorder)
    nsmOrder = getNeighbourSelectionMechanismForId(nsmOrder)

    thresholdEvaluationMethod = getThresholdEvaluationMethod(thresholdEvaluationMethod)

    if eventEffect == EventEffect.ALIGN_TO_FIXED_ANGLE:
        initialState = (None, None, None)
    else:
        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)

    events = getEvents(domainSize, radius, eventEffect)
    switchSummary = getSwitchSummary(threshold, numberOfPreviousSteps, nsmDisorder, nsmOrder, kDisorder, kOrder)

    #print(f"th={threshold}, pS={previousSteps}, doK={kDisorder}, thEM={thresholdEvaluationMethod}, oK={kOrder}, doNsm={nsmDisorder}, oNsm={nsmOrder}")
    results = []
    for i in range(numIterationsPerRun):
        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                radius=radius,
                                                noise=noise,
                                                numberOfParticles=n,
                                                k=k,
                                                neighbourSelectionMechanism=nsm,
                                                speed=speed,
                                                switchSummary=switchSummary,
                                                degreesOfVision=degreesOfVision,
                                                events=events,
                                                thresholdEvaluationMethod=thresholdEvaluationMethod)
        simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

        times, positions, orientations = simulationData

        orders = []
        for window in orderTimeWindows:
            [orders.append(ServiceMetric.computeGlobalOrder(orients)) for orients in orientations[window[0]:window[1]]]
        for window in disorderTimeWindows:
            [orders.append(1-ServiceMetric.computeGlobalOrder(orients)) for orients in orientations[window[0]:window[1]]]
        
        results.append(np.average(orders))
        # ServiceSavedModel.saveModel(simulationData=simulationData, path=f"test_{i+1}.json", 
        #                             modelParams=simulator.getParameterSummary())
    return np.average(results)
    
def getFinalOrderForDistant(threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder):
    """
    disordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    eventEffect = EventEffect.ALIGN_TO_FIXED_ANGLE
    if nsmDisorder != -1:
        nsm = nsmDisorder
    else:
        nsm = nsmOrder
    if kDisorder != -1:
        k = kDisorder
    else:
        k = kOrder
    orderTimeWindows = orderTimeWindowsInitialDisorder
    disorderTimeWindows = disorderTimeWindowsInitialDisorder
    return getFinalOrderForSwitching(eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, 
                                     previousSteps=previousSteps, thresholdEvaluationMethod=thresholdEvaluationMethod, 
                                     nsmDisorder=nsmDisorder, nsmOrder=nsmOrder, kDisorder=kDisorder, kOrder=kOrder,
                                     orderTimeWindows=orderTimeWindows, disorderTimeWindows=disorderTimeWindows)

def getFinalOrderForPredator(threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    eventEffect = EventEffect.AWAY_FROM_ORIGIN
    if nsmOrder != -1:
        nsm = nsmOrder
    else:
        nsm = nsmDisorder
    if kOrder != -1:
        k = kOrder
    else:
        k = kDisorder
    orderTimeWindows = orderTimeWindowsInitialOrder
    disorderTimeWindows = disorderTimeWindowsInitialOrder
    return getFinalOrderForSwitching(eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, 
                                     previousSteps=previousSteps, thresholdEvaluationMethod=thresholdEvaluationMethod, 
                                     nsmDisorder=nsmDisorder, nsmOrder=nsmOrder, kDisorder=kDisorder, kOrder=kOrder,
                                     orderTimeWindows=orderTimeWindows, disorderTimeWindows=disorderTimeWindows)

def getFinalOrderForRandom(threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    eventEffect = EventEffect.RANDOM
    if nsmOrder != -1:
        nsm = nsmOrder
    else:
        nsm = nsmDisorder
    if kOrder != -1:
        k = kOrder
    else:
        k = kDisorder
    orderTimeWindows = orderTimeWindowsInitialOrder
    disorderTimeWindows = disorderTimeWindowsInitialOrder
    return getFinalOrderForSwitching(eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, 
                                     previousSteps=previousSteps, thresholdEvaluationMethod=thresholdEvaluationMethod, 
                                     nsmDisorder=nsmDisorder, nsmOrder=nsmOrder, kDisorder=kDisorder, kOrder=kOrder,
                                     orderTimeWindows=orderTimeWindows, disorderTimeWindows=disorderTimeWindows)

def getOverallFinalOrder(threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder):
    comboResults = []
    comboResults.append(getFinalOrderForDistant(threshold=threshold,
                                                previousSteps=previousSteps,
                                                thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                nsmDisorder=nsmDisorder,
                                                nsmOrder=nsmOrder,
                                                kDisorder=kDisorder,
                                                kOrder=kOrder))
    comboResults.append(getFinalOrderForPredator(threshold=threshold,
                                                previousSteps=previousSteps,
                                                thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                nsmDisorder=nsmDisorder,
                                                nsmOrder=nsmOrder,
                                                kDisorder=kDisorder,
                                                kOrder=kOrder))
    comboResults.append(getFinalOrderForRandom(threshold=threshold,
                                                previousSteps=previousSteps,
                                                thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                nsmDisorder=nsmDisorder,
                                                nsmOrder=nsmOrder,
                                                kDisorder=kDisorder,
                                                kOrder=kOrder))

    return np.average(np.array(comboResults))

def fitness_function(params):
    threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder = params
    previousSteps = int(previousSteps)
    thresholdEvaluationMethod = int(thresholdEvaluationMethod)
    nsmDisorder = int(nsmDisorder)
    nsmOrder = int(nsmOrder)
    kDisorder = int(kDisorder)
    kOrder = int(kOrder)

    # cannot opt to ignore all values for k
    if kOrder == -1 and kDisorder == -1:
        return 0
    
    return getOverallFinalOrder(threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder)

# Create the initial population
def create_initial_population(size, lower_bounds, upper_bounds):
    population = []
    for _ in range(size):
        individual = (random.uniform(lower_bounds[0], upper_bounds[0]),
                      random.uniform(lower_bounds[1], upper_bounds[1]),
                      random.choice(range(lower_bounds[2], upper_bounds[2]+1)),
                      random.choice(range(lower_bounds[3], upper_bounds[3]+1)),
                      random.choice(range(lower_bounds[4], upper_bounds[4]+1)),
                      random.choice(range(lower_bounds[5], upper_bounds[5]+1)),
                      random.choice(range(lower_bounds[6], upper_bounds[6]+1)))       
        population.append(individual)
    return population

# Selection function using tournament selection
def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Crossover function
def crossover(parent1, parent2):
    alpha = random.random()
    child1 = tuple(alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2))
    child2 = tuple(alpha * p2 + (1 - alpha) * p1 for p1, p2 in zip(parent1, parent2))
    return child1, child2

# Mutation function
def mutation(individual, mutation_rate, lower_bounds, upper_bounds):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutation_amount = random.uniform(-1, 1)
            individual[i] += mutation_amount
            # Ensure the individual stays within bounds
            individual[i] = max(min(individual[i], upper_bounds[i]), lower_bounds[i])
    return tuple(individual)

# Main genetic algorithm function
def genetic_algorithm(population_size, lower_bounds, upper_bounds, generations, mutation_rate):
    population = create_initial_population(population_size, lower_bounds, upper_bounds)
    
    # Prepare for plotting
    fig, axs = plt.subplots(7, 1, figsize=(12, 18))  # 7 rows, 1 column for subplots
    best_performers = []
    all_populations = []
    all_fitnesses = []

    # Prepare for table
    table = PrettyTable()
    #table.field_names = ["Generation", "a", "b", "c", "Fitness"]
    
    table.field_names = ["Generation", "Threshold", "Previous steps", "TEM", "Do Nsm", "O Nsm", "Do k", "O k", "Fitness"]

    for generation in range(generations):
        print(f"gen {generation+1}/{generations}")
        
        fitnesses = {ind: fitness_function(ind) for ind in population}
        print("fitnesses:")
        print(fitnesses)
        all_fitnesses.append(fitnesses)

        # Store the best performer of the current generation
        best_individual = max(fitnesses, key=fitnesses.get)
        best_fitness = fitnesses[best_individual]
        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        #table.add_row([generation + 1, best_individual[0], best_individual[1], best_individual[2], best_fitness])
        table.add_row([generation + 1, best_individual[0], best_individual[1], best_individual[2], best_individual[3], best_individual[4], best_individual[5], best_individual[6], best_fitness])

        population = selection(population, fitnesses)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            child1, child2 = crossover(parent1, parent2)

            next_population.append(mutation(child1, mutation_rate, lower_bounds, upper_bounds))
            next_population.append(mutation(child2, mutation_rate, lower_bounds, upper_bounds))

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population

    # Print the table
    print(table)

    dict = {"all_populations": all_populations, "all_fitnesses": all_fitnesses, "best_individual": best_individual, "best_performers": best_performers}
    ServiceSavedModel.saveGenInfo("genetic_choice.json", dict)

    # Plot the population of one generation (last generation)
    final_population = all_populations[-1]
    final_fitnesses = all_fitnesses[-1]

    axs[0].scatter(range(len(final_population)), [ind[0] for ind in final_population], color='blue', label='threshold')
    axs[0].scatter([final_population.index(best_individual)], [best_individual[0]], color='cyan', s=100, label='Best Individual: Threshold')
    axs[0].set_ylabel('Threshold', color='blue')
    axs[0].legend(loc='upper left')
    
    axs[1].scatter(range(len(final_population)), [ind[1] for ind in final_population], color='green', label='Previous steps')
    axs[1].scatter([final_population.index(best_individual)], [best_individual[1]], color='magenta', s=100, label='Best Individual: Previous steps')
    axs[1].set_ylabel('Previous steps', color='green')
    axs[1].legend(loc='upper left')
    
    axs[2].scatter(range(len(final_population)), [ind[2] for ind in final_population], color='red', label='Threshold evaluation method')
    axs[2].scatter([final_population.index(best_individual)], [best_individual[2]], color='yellow', s=100, label='Best Individual: Threshold evaluation method')
    axs[2].set_ylabel('TEM', color='red')
    axs[2].set_xlabel('Individual Index')
    axs[2].legend(loc='upper left')
        
    axs[3].scatter(range(len(final_population)), [ind[3] for ind in final_population], color='blue', label='Disorder neighbour selection mechanism')
    axs[3].scatter([final_population.index(best_individual)], [best_individual[3]], color='cyan', s=100, label='Best Individual: Disorder neighbour selection mechanism')
    axs[3].set_ylabel('Do nsm', color='blue')
    axs[3].set_xlabel('Individual Index')
    axs[3].legend(loc='upper left')
           
    axs[4].scatter(range(len(final_population)), [ind[4] for ind in final_population], color='green', label='Order neighbour selection mechanism')
    axs[4].scatter([final_population.index(best_individual)], [best_individual[4]], color='magenta', s=100, label='Best Individual: Order neighbour selection mechanism')
    axs[4].set_ylabel('O nsm', color='green')
    axs[4].set_xlabel('Individual Index')
    axs[4].legend(loc='upper left') 

    axs[5].scatter(range(len(final_population)), [ind[5] for ind in final_population], color='red', label='Disorder k')
    axs[5].scatter([final_population.index(best_individual)], [best_individual[5]], color='yellow', s=100, label='Best Individual: Disorder k')
    axs[5].set_ylabel('Do k', color='red')
    axs[5].set_xlabel('Individual Index')
    axs[5].legend(loc='upper left') 

    axs[6].scatter(range(len(final_population)), [ind[6] for ind in final_population], color='blue', label='Order k')
    axs[6].scatter([final_population.index(best_individual)], [best_individual[6]], color='cyan', s=100, label='Best Individual: Order k')
    axs[6].set_ylabel('O k', color='blue')
    axs[6].set_xlabel('Individual Index')
    axs[6].legend(loc='upper left') 
    
    axs[0].set_title(f'Final Generation ({generations}) Population Solutions')
    plt.savefig(f"genetic_threshold_previousSteps_d={density}_r={radius}_one_generation.svg")


    # Plot the values of a, b, and c over generations
    generations_list = range(1, len(best_performers) + 1)
    threshold_values = [ind[0][0] for ind in best_performers]
    previousSteps_values = [(int(ind[0][1])/tmax) for ind in best_performers]
    thresholdEvaluationMethod_values = [int(ind[0][2]) for ind in best_performers]
    doNsm_values = [int(ind[0][3]) for ind in best_performers]
    oNsm_values = [int(ind[0][4]) for ind in best_performers]
    doK_values = [int(ind[0][5]) for ind in best_performers]
    oK_values = [int(ind[0][6]) for ind in best_performers]

    """
    c_values = [ind[0][2] for ind in best_performers]
    """
    fig, ax = plt.subplots()
    ax.plot(generations_list, threshold_values, label='threshold', color='blue')
    ax.plot(generations_list, previousSteps_values, label='previous steps', color='green')
    ax.plot(generations_list, thresholdEvaluationMethod_values, label='TEM', color='cyan')
    ax.plot(generations_list, doNsm_values, label='do nsm', color='red')
    ax.plot(generations_list, oNsm_values, label='o nsm', color='orange')
    ax.plot(generations_list, doK_values, label='do k', color='darkviolet')
    ax.plot(generations_list, oK_values, label='o k', color='fuchsia')
    """
    ax.plot(generations_list, c_values, label='c', color='red')
    """
    ax.set_xlabel('Generation')
    ax.set_ylabel('Parameter Values')
    ax.set_title('Parameter Values Over Generations')
    ax.legend()
    plt.savefig(f"genetic_threshold_previousSteps_d={density}_r={radius}_abc_generations.svg")


    # Plot the fitness values over generations
    best_fitness_values = [fit[1] for fit in best_performers]
    min_fitness_values = [min(fitnesses.values()) for fitnesses in all_fitnesses]
    max_fitness_values = [max(fitnesses.values()) for fitnesses in all_fitnesses]
    fig, ax = plt.subplots()
    ax.plot(generations_list, best_fitness_values, label='Best Fitness', color='black')
    ax.fill_between(generations_list, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Over Generations')
    ax.legend()
    plt.savefig(f"genetic_threshold_previousSteps_d={density}_r={radius}_fitness_over_generations.svg")


    """
    # Plot the quadratic function for each generation
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, generations))
    for i, (best_ind, best_fit) in enumerate(best_performers):
        color = colors[i]
        
        # a, b, c = best_ind
        # x_range = np.linspace(lower_bound, upper_bound, 400)
        # y_values = a * (x_range ** 2) + b * x_range + c
        
        threshold, previousSteps = best_ind
        x_range = np.linspace(lower_bounds, upper_bounds, 400)
        y_values = threshold * (x_range ** 2) + previousSteps * x_range
        ax.plot(x_range, y_values, color=color)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Quadratic Function')

    # Create a subplot for the colorbar
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    norm = plt.cm.colors.Normalize(vmin=0, vmax=generations)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, cax=cax, orientation='vertical', label='Generation')

    plt.savefig(f"genetic_threshold_previousSteps_d={density}_r={radius}_quadratic function.svg")
    """
    plt.show()

    return max(fitnesses, key=fitnesses.get)

# Run the genetic algorithm
best_solution = genetic_algorithm(population_size, lower_bounds, upper_bounds, generations, mutation_rate)
#print(f"Best solution found: a = {best_solution[0]}, b = {best_solution[1]}, c = {best_solution[2]}")
print(f"Best solution found: threshold = {best_solution[0]}, previous steps = {best_solution[1]}, thresholdEvaluationMethod = {best_solution[2]}, nsmDisorder = {best_solution[3]}, nsmOrder = {best_solution[4]}, kDisorder = {best_solution[5]}, kOrder = {best_solution[6]}")
#threshold, previousSteps, thresholdEvaluationMethod, nsmDisorder, nsmOrder, kDisorder, kOrder
