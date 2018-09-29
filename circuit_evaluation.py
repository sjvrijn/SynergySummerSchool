from enum import Enum
import pyrtl
from collections import namedtuple

__all__ = ['Gate', 'Operand', 'evaluate_circuit']

Gate = namedtuple('Gate', ['inputs', 'operand'])


class Operand(Enum):
    NOT = 0
    AND = 1
    OR = 2
    NAND = 3
    NOR = 4
    XOR = 5
    MEM = 6
    PASS = 7


def not_gate(a):
    return ~a

def and_gate(a, b):
    return a & b

def or_gate(a, b):
    return a | b

def nand_gate(a, b):
    return ~(a & b)

def nor_gate(a, b):
    return ~(a | b)

def xor_gate(a, b):
    return a ^ b

def mem_gate(mem, a):
    mem.next <<= a
    return mem

def pass_gate(a):
    return a


def translate(gate, inputs, num_memories, mem_idx=None):
    idx_a, idx_b = gate.inputs
    input_a = inputs[idx_a + num_memories]
    input_b = inputs[idx_b + num_memories]

    try:
        if gate.operand == Operand.NOT:
            return not_gate(input_a)

        elif gate.operand == Operand.AND:
            return and_gate(input_a, input_b)

        elif gate.operand == Operand.OR:
            return or_gate(input_a, input_b)

        elif gate.operand == Operand.NAND:
            return nand_gate(input_a, input_b)

        elif gate.operand == Operand.NOR:
            return nor_gate(input_a, input_b)

        elif gate.operand == Operand.XOR:
            return xor_gate(input_a, input_b)

        elif gate.operand == Operand.MEM:
            return mem_gate(inputs[mem_idx], input_a)

        elif gate.operand == Operand.PASS:
            return pass_gate(input_a)

        else:
            raise ValueError(f'invalid gate operand {gate.operand}')

    except IndexError:
        raise Exception(f'Invalid input index detected: {idx_a} or {idx_b}')


def mem_scan(matrix):

    memories = []
    for column in matrix:
        for gate in column:
            if gate.operand == Operand.MEM:
                memories.append(pyrtl.wire.Register(1))
    return list(reversed(memories))


def evaluate_circuit(matrix, input_sequence, expected_output):
    pyrtl.reset_working_block()
    n_inputs = len(input_sequence[0])
    global_inputs = [pyrtl.Input(1, str(inp)) for inp in range(n_inputs)]
    inputs = mem_scan(matrix)
    num_memories = len(inputs)
    cur_memory = num_memories - 1
    inputs.extend(global_inputs)

    gate_matrix = translate_matrix_to_pyrtl(cur_memory, inputs, matrix, num_memories)
    outputs = connect_outputs(gate_matrix, matrix)
    output_bits = simulate_circuit(input_sequence, inputs, outputs, n_inputs, num_memories)

    return calculate_correctness(expected_output, output_bits)


def translate_matrix_to_pyrtl(cur_memory, inputs, matrix, num_memories):

    gate_matrix = []
    for column in matrix:
        gate_column = []
        gate_matrix.append(gate_column)
        for gate in column:

            gate_func = translate(gate, inputs, num_memories=num_memories, mem_idx=cur_memory)
            if gate.operand == Operand.MEM:
                cur_memory -= 1
            gate_column.append(gate_func)
        inputs.extend(gate_column)
    return gate_matrix


def connect_outputs(gate_matrix, matrix):
    outputs = [pyrtl.Output(1, f'out_{i}') for i in range(len(matrix[-1]))]
    for out, gate in zip(outputs, gate_matrix[-1]):
        out <<= gate
    return outputs


def simulate_circuit(input_bits, inputs, outputs, n_inputs, num_memories):
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)
    output = []
    for bits in input_bits:
        bits_in = {inp.name: bit for inp, bit in zip(inputs[num_memories:num_memories + n_inputs], bits)}
        sim.step(bits_in)
        output.append([sim.inspect(out) for out in outputs])
    return output


def calculate_correctness(expected, actual):
    """
    Returns the percentage of integers in actual that match the ones in expected

    Actual and Expected should be 2D lists
    """

    # flatten lists
    expected = [item for sublist in expected for item in sublist]
    actual = [item for sublist in actual for item in sublist]

    total = len(actual)
    matches = [exp == act for (exp, act) in zip(expected, actual)]
    return sum(matches) / total
