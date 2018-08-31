from circuit_evaluation import Gate, Operand, evalute_circuit
from itertools import product

import random

from deap import base
from deap import creator
from deap import tools

from deap import algorithms

def int_tuple_to_Gate(int_tuple):
    input1, input2, operand = int_tuple
    return Gate([input1, input2], Operand(operand))



def int_list_to_gates(int_list, shape):

    matrix = [[] for _ in range(shape[1])]
    matrix_indices = product(range(shape[1]), range(shape[0]))

    for int_tuple, index in zip(int_list, matrix_indices):
        col, row = index
        matrix[col][row] = int_tuple_to_Gate(int_tuple)

    return matrix



def create_matrix(individual, n, n_inputs, n_operators):
    container = []
    for x in range(n):
        col = []
        for y in range(n):
            container.append( (random.randint(0,n_inputs + n*x), random.randint(0,n_inputs + n*x), random.randint(0,n_operators)) )
    ind = individual(container)

    return ind


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)



MATRIX_DIM=3
N_INPUTS = 4
N_OPERATORS = 3
POP_SIZE = 100


toolbox = base.Toolbox()
toolbox.register("attr_int", random.random)
toolbox.register("individual", create_matrix, creator.Individual,
                  n=MATRIX_DIM, n_inputs=N_INPUTS, n_operators=N_OPERATORS)


toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(POP_SIZE=100)


#TODO: Fitness function
def evalOneMax(individual):
    #TODO: evaluate function
    return (sum(individual),)

def crossOver(ind1, ind2):
    #don't do crossover
    #the crossover rate should be zero anyway
    return ind1, ind2

def mutate(ind, n_operators):
    #swap operator to random one
    ind[2]= random.randint(0, n_operators)
    return ind


toolbox.register("mutate", mutate, n_operators=N_OPERATORS)
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", crossOver)
toolbox.register("select", tools.selTournament, tournsize=3)

#cxbp = crossover chance
#mztpb = mutation rate
#ngen = number of generations
logbook = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.2, ngen=1000)

#pop now has the final population
for ind in pop:
    print(ind)
    print(ind.fitness.values)
    print()

