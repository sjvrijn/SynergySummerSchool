from circuit_evaluation import Gate, evalute_circuit


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
            Gate([0,1], 'AND'),
            Gate([2,-1], 'OR'),
        ],
        [
            Gate([3,9], 'MEM'),
            Gate([3,4], 'OR'),
        ],
    ]
    print(evalute_circuit(3, matrix, input_bits, expected_output))


if __name__ == '__main__':
    example()