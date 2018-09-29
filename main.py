from circuit_evaluation import Gate, Operand, evaluate_circuit


def example():
    input_bits = [
        [0,0,0],
        [0,0,1],
        [0,0,0],
        [1,1,1],
        [0,0,0],
        [0,0,0],
        [0,0,1],
    ]

    expected_output = [
        [0,0],
        [0,1],
        [0,0],
        [0,1],
        [1,1],
        [0,0],
        [0,1],
    ]

    matrix = [
        [
            Gate([0,1], Operand.AND),
            Gate([2,-1], Operand.OR),
        ],
        [
            Gate([3,0], Operand.MEM),
            Gate([3,4], Operand.OR),
        ],
    ]
    print(evaluate_circuit(matrix=matrix, input_sequence=input_bits, expected_output=expected_output))


if __name__ == '__main__':
    example()
