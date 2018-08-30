import pyrtl
from collections import namedtuple
from random import getrandbits


Gate = namedtuple('Gate', ['inputs', 'operand'])


def and_gate(a, b):
    return a & b

def or_gate(a, b):
    return a | b


def translate(gate, inputs):
    try:
        if gate.operand == 'and':
            return and_gate(inputs[gate.inputs[0]], inputs[gate.inputs[1]])
        elif gate.operand == 'or':
            return or_gate(inputs[gate.inputs[0]], inputs[gate.inputs[1]])
        else:
            raise ValueError(f'invalid gate operand {gate.operand}')
    except IndexError:
        raise Exception('Invalid input index detected!')

def translation(n_inputs, matrix):

    global_inputs = [pyrtl.Input(1, str(inp)) for inp in range(n_inputs)]
    inputs = []
    inputs.extend(global_inputs)
    gate_matrix = []

    for column in matrix:
        gate_column = []
        gate_matrix.append(gate_column)
        for gate in column:
            gate_func = translate(gate, inputs)
            gate_column.append(gate_func)
        inputs.extend(gate_column)

    outputs = [pyrtl.Output(1, f'out_{i}') for i in range(len(matrix[-1]))]
    for out, gate in zip(outputs, gate_matrix[-1]):
        out <<= gate

    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    for cycle in range(5):
        bits_in = {inp.name: getrandbits(1) for inp in inputs[:n_inputs]}
        sim.step(bits_in)
        print(bits_in)
        print([sim.inspect(out) for out in outputs])
        print()


matrix = [
    [
        Gate([0,1], 'and'),
        Gate([2,3], 'or'),
    ],
    [
        Gate([4,5], 'or'),
        Gate([4,5], 'and'),
    ],
]

translation(4,matrix)
