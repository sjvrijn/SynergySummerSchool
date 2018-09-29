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
    """
    Translate a `Gate` to pyrtl gate/connections
    """
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
    """
    Scan the matrix of `Gate`s and create the registers.
    """

    memories = []
    for column in matrix:
        for gate in column:
            if gate.operand == Operand.MEM:
                memories.append(pyrtl.wire.Register(1))
    return list(reversed(memories))


def evaluate_circuit(matrix, input_sequence, expected_output):
    """
    Given a matrix of `Gate`s, evaluate how well `expected_output` matches the
    actual output of the created circuit when given `input_sequence` as input.
    """
    # Initialize the circuit
    pyrtl.reset_working_block()

    # Create the initial 'connection points': inputs and registers
    circuit_inputs = [pyrtl.Input(1, str(inp)) for inp in range(len(input_sequence[0]))]
    registers = mem_scan(matrix)
    connection_points = [*registers, *circuit_inputs]

    # Create the actual pyrtl circuit and link the outputs
    gate_matrix = translate_matrix_to_pyrtl(connection_points, matrix, len(registers))
    circuit_outputs = connect_outputs(gate_matrix, matrix)

    # Simulate and return correctness value between [0,1]
    output_bits = simulate_circuit(circuit_inputs, input_sequence, circuit_outputs)
    return calculate_correctness(expected_output, output_bits)


def translate_matrix_to_pyrtl(inputs, matrix, num_registers):
    """
    Given a matrix of `Gate`s, create the actual pyrtl circuit
    """
    current_register_idx = num_registers-1

    gate_matrix = []
    for column in matrix:
        gate_column = []
        gate_matrix.append(gate_column)
        for gate in column:
            gate_func = translate(gate, inputs, num_memories=num_registers, mem_idx=current_register_idx)
            if gate.operand == Operand.MEM:
                current_register_idx -= 1
            gate_column.append(gate_func)
        inputs.extend(gate_column)
    return gate_matrix


def connect_outputs(gate_matrix, matrix):
    """
    Create and connect `pyrtl.Output`s for each of the gates in the last column
    of the circuit matrix to serve as circuit outputs.
    """
    outputs = [pyrtl.Output(1, f'out_{i}') for i in range(len(matrix[-1]))]
    for out, gate in zip(outputs, gate_matrix[-1]):
        out <<= gate
    return outputs


def simulate_circuit(circuit_inputs, input_bits, circuit_outputs):
    """
    Simulate and record the circuit output by passing in `input_bits`.
    """
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)
    output_bits = []
    for bits in input_bits:
        bits_in = {inp.name: bit for inp, bit in zip(circuit_inputs, bits)}
        sim.step(bits_in)
        output_bits.append([sim.inspect(out) for out in circuit_outputs])
    return output_bits


def calculate_correctness(expected, actual):
    """
    Returns the percentage of values in `actual` that match those in `expected`.
    `actual` and `expected` should be 2D lists.
    """

    # flatten lists
    expected = [item for sublist in expected for item in sublist]
    actual = [item for sublist in actual for item in sublist]

    total = len(actual)
    matches = [exp == act for (exp, act) in zip(expected, actual)]
    return sum(matches) / total
