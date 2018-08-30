from translator import translate

def calculate_fitness(expected, actual):
	"""
	Returns the percentage of integers in actual that match the ones in expected

	Actual and Expected should be 2D lists
	"""

	#flatten lists
	expected = [item for sublist in expected for item in sublist]
	actual = [item for sublist in actual for item in sublist]
	
	total = len(actual)
	matches = [exp == act for (exp, act) in zip(expected, actual)]
	return sum(matches)/total


def calculate_circuit_fitness(num_inputs, circuit, expected_outputs):
	"""
	Takes a matrix representing the circuit, evaluates it, and returns the fitness
	"""
	outputs = translate(num_inputs, circuit)
	return calculate_fitness(expected_outputs, outputs)


