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


def random_gate(n_inputs, n_operators):
    return (random.randint(0, n_inputs-1),
            random.randint(0, n_inputs-1),
            random.randint(0, n_operators-1))


def initialize_random_matrix(Individual, shape, n_inputs, n_operators):
    """Create a random matrix that represents a valid circuit"""

    cols, rows = shape
    container = []
    for x in range(cols):
        container.extend([random_gate(n_inputs, n_operators) for _ in range(rows)])
        n_inputs += rows

    return Individual(container)


#Define genetic operators
def evaluate(individual, shape, input_sequence, output_sequence):
    #TODO: evaluate function, has to retrun a tuple with a single member: the fitness
    matrix = int_list_to_gates(individual, shape)
    e = evaluate_circuit(matrix, input_sequence, output_sequence)
    return (e,)


def crossOver(ind1, ind2):
    #don't do crossover
    #the crossover rate should be zero anyway
    return ind1, ind2


def mutate(individual, n_operators, n_inputs, op_mutation_rate, input_mutation_rate):
    #swap operator to random one

    # mutate operands first and count how many memories we have
    num_memories = 0
    for idx, gate_tuple in enumerate(individual):
        input1, input2, operand = gate_tuple
        if random.random() < op_mutation_rate:
            operand = random.randint(0, n_operators-1)
        if Operand(operand) == Operand.MEM:
            num_memories += 1
        individual[idx] = (input1, input2, operand)

    min_input = -num_memories
    max_input = n_inputs - 1

    # Now we mutate the inputs based on the available gates and global inputs
    for idx, gate_tuple in enumerate(individual):
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
        individual[idx] = (input1, input2, operand)

    return (individual,)



if __name__ == '__main__':
    # Example test-case requiring at least one Register
    input_sequence = [
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
    output_sequence = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
    ]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


    #Population parameters
    MATRIX_SHAPE = (2, 2)
    N_INPUTS = 1
    N_OPERATORS = len(Operand)
    POP_SIZE = 20
    N_GENERATIONS = 200

    # NOTE: these two separate mutation rates currently over-ride the DEAP-mutation rate
    op_mutation_rate = 0.25
    input_mutation_rate = 0.125


    toolbox = base.Toolbox()

    #Register an individual: a list of tuples, with each tuple representing a gate
    toolbox.register("attr_int", random.random)
    toolbox.register("individual", initialize_random_matrix, creator.Individual,
                     shape=MATRIX_SHAPE, n_inputs=N_INPUTS, n_operators=N_OPERATORS)

    #Create a population: list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(POP_SIZE)

    #setup the algorithm: link the above functions, set selection strategy
    toolbox.register("mutate", mutate, n_operators=N_OPERATORS, n_inputs=N_INPUTS,
                     op_mutation_rate=op_mutation_rate, input_mutation_rate=input_mutation_rate)
    toolbox.register("evaluate", evaluate, shape=MATRIX_SHAPE,
                     input_sequence=input_sequence, output_sequence=output_sequence)
    toolbox.register("mate", crossOver)
    toolbox.register("select", tools.selTournament, tournsize=3)


    #start the simple builtin algorithm
    #cxbp = crossover chance
    #mutpb = mutation rate
    #ngen = number of generations
    logbook = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.2, ngen=N_GENERATIONS)

    #pop now has the final population
    for ind in pop:
        print(ind)
        print(ind.fitness.values)
        print()


