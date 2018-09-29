import random
from itertools import product
from deap import algorithms, base, creator, tools
from circuit_evaluation import Gate, Operand, evaluate_circuit


def int_tuple_to_Gate(int_tuple):
    input1, input2, operand = int_tuple
    return Gate([input1, input2], Operand(operand))


def int_list_to_gates(int_list, shape):
    """Create a matrix of Gates out of a pure list of integer tuples"""
    gate_matrix = [[None for _ in range(shape[0])] for _ in range(shape[1])]
    matrix_indices = product(range(shape[1]), range(shape[0]))

    for int_tuple, index in zip(int_list, matrix_indices):
        col, row = index
        gate_matrix[col][row] = int_tuple_to_Gate(int_tuple)

    return gate_matrix


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


def crossover(ind1, ind2):
    """Crossover function can be implemented here"""
    return ind1, ind2


def mutate_sequential_circuit(individual, shape, n_operators, n_inputs, op_mutation_rate, input_mutation_rate):
    """
    Perform the mutation of a sequential circuit representation.

    This is done in several steps:
    1. First the operands of the gates in the circuits are mutated.
       The number of Registers (after mutation) is simultaneously counted,
       as we need this number to determine the valid Gate input numbers.

    2. The input connections are mutated next. This can happen either
       because it is randomly chosen to be mutated, or otherwise because
       it was connected to a Register that no longer exists.

    NOTE: THIS IS AN ARBITRARY DESIGN DECISION FOR THE MUTATION OPERATOR
          IT MAY BE WORHTWHILE CONSIDERING ALTERNATIVES, SUCH AS
          EXPLICITLY RECONNECTING IT TO ANOTHER REGISTER RATHER THAN
          JUST ANY RANDOM AVAILABLE SIGNAL
    """
    num_cols, num_rows = shape

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

        if idx > 0 and idx % num_rows == 0:
            max_input += num_rows

        if random.random() < input_mutation_rate or input1 < min_input:
            input1 = random.randint(min_input, max_input)
        if random.random() < input_mutation_rate or input2 < min_input:
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

    crossover_rate = 0
    mutation_rate = 1 / (MATRIX_SHAPE[0] * MATRIX_SHAPE[1])
    op_mutation_rate = 1/3 * mutation_rate
    input_mutation_rate = 1/3 * mutation_rate

    toolbox = base.Toolbox()

    #Register an individual: a list of tuples, with each tuple representing a gate
    toolbox.register("individual", initialize_random_matrix, creator.Individual,
                     shape=MATRIX_SHAPE, n_inputs=N_INPUTS, n_operators=N_OPERATORS)

    #Create a population: list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(POP_SIZE)

    #setup the algorithm: link the above functions, set selection strategy
    toolbox.register("mutate", mutate_sequential_circuit, shape=MATRIX_SHAPE,
                     n_operators=N_OPERATORS, n_inputs=N_INPUTS,
                     op_mutation_rate=op_mutation_rate, input_mutation_rate=input_mutation_rate)
    toolbox.register("evaluate", evaluate, shape=MATRIX_SHAPE,
                     input_sequence=input_sequence, output_sequence=output_sequence)
    toolbox.register("mate", crossover)
    toolbox.register("select", tools.selTournament, tournsize=3)


    logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate, ngen=N_GENERATIONS)

    #pop now has the final population
    for idx, ind in enumerate(pop):
        print(f'{round(ind.fitness.values[0]*100, ndigits=1)}% :', ind)

