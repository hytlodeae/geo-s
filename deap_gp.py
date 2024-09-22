# DEAP Genetic Programming


# Importing framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools, gp

# Importing other libraries
import random
import numpy as np
import operator
import time
import os

# Creates a directory to save the results
# This is the same as the code from Karine but I don't see how this could be an issue
experiment_name = "gp_optimization_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
n_neurons = 100
env = Environment(
    experiment_name=experiment_name,
    player_controller=player_controller(
        n_neurons
    ),  ##not sure about this will it require us to change demo_controller bc we are using a tree????/
    speed="normal",
    enemies=[4],
    sound="off",
    logs="off",
    savelogs="no",
)


env.state_to_log()  # Env state # from karine

"""' import multiprocessing

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

# Continue on with the evolutionary algorithm
pool.close()"""

toolbox = base.Toolbox()

# Creating the primitive nodes for the tree
pset = gp.PrimitiveSet("MAIN", arity=2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
# Adding the terminal nodes
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))  # Random constants


# DCreating the fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
creator.create(
    "Individual", gp.PrimitiveTree, fitness=creator.FitnessMax
)  # This sets up individuals as trees that need to be evolved
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)


def evaluate(x):
    return np.array([env.play(pcont=y)[0] for y in x])


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

ngen = 10
cxpb = 0.3  # Probability of a cross-over happening
mutpb = 0.3  # Probability of a mutation happening
pop_size = 100
pop = toolbox.population(n=pop_size)
initial_fit = toolbox.evaluate(pop)  # The fitness of the first gen after playing once
for ind, fit in zip(pop, initial_fit):  #
    ind.fitness.values = (fit,)

for g in range(ngen):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)

    # The population is entirely replaced by the offspring
    pop[:] = offspring
