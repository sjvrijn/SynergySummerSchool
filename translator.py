import pyrtl
from collections import namedtuple


Gate = namedtuple('Gate', ['inputs', 'operand'])


def and_gate(a, b):
    return a & b

def or_gate(a, b):
    return a | b

def mem_gate(mem, a):
    mem.next <<= a
    return mem


def translate(gate, inputs, num_memories, mem_idx=None):
    try:
        if gate.operand == 'and':
            return and_gate(inputs[gate.inputs[0]+num_memories], inputs[gate.inputs[1]+num_memories])
        elif gate.operand == 'or':
            return or_gate(inputs[gate.inputs[0]+num_memories], inputs[gate.inputs[1]+num_memories])
        elif gate.operand == 'mem':
            return mem_gate(inputs[mem_idx], inputs[gate.inputs[0]+num_memories])
        else:
            raise ValueError(f'invalid gate operand {gate.operand}')
    except IndexError:
        raise Exception('Invalid input index detected!')


def mem_scan(matrix):

    memories = []
    for column in matrix:
        for gate in column:
            if gate.operand == 'mem':
                memories.append(pyrtl.wire.Register(1))
    return list(reversed(memories))


def translation(n_inputs, matrix):

    global_inputs = [pyrtl.Input(1, str(inp)) for inp in range(n_inputs)]
    inputs = mem_scan(matrix)
    num_memories = len(inputs)
    cur_memory = num_memories - 1
    inputs.extend(global_inputs)
    gate_matrix = []

    for column in matrix:
        gate_column = []
        gate_matrix.append(gate_column)
        for gate in column:
            gate_func = translate(gate, inputs, num_memories=num_memories, mem_idx=cur_memory)
            if gate.operand == 'mem':
                cur_memory -= 1
            gate_column.append(gate_func)
        inputs.extend(gate_column)

    outputs = [pyrtl.Output(1, f'out_{i}') for i in range(len(matrix[-1]))]
    for out, gate in zip(outputs, gate_matrix[-1]):
        out <<= gate

    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    input_bits = ['000',
                  '001',
                  '000',
                  '111',
                  '000',
                  '001']

    for bits in input_bits:

        bits_in = {inp.name: int(bit) for inp, bit in zip(inputs[num_memories:num_memories+n_inputs], bits)}
        sim.step(bits_in)
        print(f'{bits_in} --> {[sim.inspect(out) for out in outputs]}')


matrix = [
    [
        Gate([0,1], 'and'),
        Gate([2,-1], 'or'),
    ],
    [
        Gate([3,9], 'mem'),
        Gate([3,4], 'or'),
    ],
]

translation(3,matrix)
