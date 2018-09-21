import random
from itertools import product
from deap import algorithms, base, creator, tools
from circuit_evaluation import Gate, Operand, evaluate_circuit


def int_tuple_to_Gate(int_tuple):
    input1, input2, operand = int_tuple
    return Gate([input1, input2], Operand(operand))


def int_list_to_gates(int_list, shape):

    matrix = [[None for _ in range(shape[0])] for _ in range(shape[1])]
    matrix_indices = product(range(shape[1]), range(shape[0]))

    for int_tuple, index in zip(int_list, matrix_indices):
        col, row = index
        matrix[col][row] = int_tuple_to_Gate(int_tuple)

    return matrix


def initialize_random_matrix(individual, shape, n_inputs, n_operators):
    """Create a random matrix that represents a valid circuit"""

    cols, rows = shape
    container = []
    for x in range(cols):
        for y in range(rows):
            container.append( (random.randint(0, n_inputs-1),
                               random.randint(0, n_inputs-1),
                               random.randint(0, n_operators-1)) )
        n_inputs += rows
    ind = individual(container)

    return ind


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


#Population parameters
MATRIX_SHAPE = (2, 2)
N_INPUTS = 1
N_OPERATORS = len(Operand)
POP_SIZE = 20
op_mutation_rate = 0.25
input_mutation_rate = 0.125


#Register an individual: a list of tuples, with each tuple representing a gate
toolbox = base.Toolbox()
toolbox.register("attr_int", random.random)
toolbox.register("individual", initialize_random_matrix, creator.Individual,
                 shape=MATRIX_SHAPE, n_inputs=N_INPUTS, n_operators=N_OPERATORS)


#Create a population: list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(POP_SIZE)



n_inputs = 1
n_outputs = 2
input_bits = [
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [0],
    [1],
    [0],
    [0],
]
output_bits = [
    [0,0],
    [0,0],
    [0,0],
    [0,0],
    [0,0],
    [1,1],
    [0,0],
    [0,0],
    [1,1],
    [1,1],
    [0,0],
    [0,0],
    [0,0],
]


#Define genetic operators
def evaluate(individual):
    #TODO: evaluate function, has to retrun a tuple with a single member: the fitness
    matrix = int_list_to_gates(individual, MATRIX_SHAPE)
    e = evaluate_circuit(n_inputs, matrix, input_bits, output_bits)
    return (e,)


def crossOver(ind1, ind2):
    #don't do crossover
    #the crossover rate should be zero anyway
    return ind1, ind2


def mutate(ind, n_operators):
    #swap operator to random one

    # mutate operands first and count how many memories we have
    num_memories = 0
    for idx, gate_tuple in enumerate(ind):
        input1, input2, operand = gate_tuple
        if random.random() < op_mutation_rate:
            operand = random.randint(0, n_operators-1)
        if Operand(operand) == Operand.MEM:
            num_memories += 1
        ind[idx] = (input1, input2, operand)

    min_input = -num_memories
    max_input = n_inputs - 1

    # Now we mutate the inputs based on the available gates and global inputs
    for idx, gate_tuple in enumerate(ind):
        input1, input2, operand = gate_tuple
        # First we repair any indices that may have become broken by memory that was removed
        if input1 < min_input:
            input1 = random.randint(min_input, max_input)
        if input2 < min_input:
            input2 = random.randint(min_input, max_input)

        # Now we actually mutate
        if random.random() < input_mutation_rate:
            input1 = random.randint(min_input, max_input)
        if random.random() < input_mutation_rate:
            input2 = random.randint(min_input, max_input)
        ind[idx] = (input1, input2, operand)

    return (ind,)


#setup the algorithm: link the above functions, set selection strategy
toolbox.register("mutate", mutate, n_operators=N_OPERATORS)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossOver)
toolbox.register("select", tools.selTournament, tournsize=3)


#start the simple builtin algorithm
#cxbp = crossover chance
#mztpb = mutation rate
#ngen = number of generations
logbook = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.2, ngen=100)

#pop now has the final population
for ind in pop:
    print(ind)
    print(ind.fitness.values)
    print()


