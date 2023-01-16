import numpy as np
from metaheuristics.geneticalgorithm import GeneticAlgorithm


class GeneticAlgorithmForTravelingSalesman(GeneticAlgorithm):
    def __init__(
        self,
        problem,
        param_mutation_probability=0.02,
        param_population_size=100,
        param_max_number_iterations=10,
        param_logging=False
    ):
        super().__init__(problem, param_population_size,
                         param_max_number_iterations, param_logging)
        self._param_mutation_probability = param_mutation_probability

    def crossover(self, parents):
        """Create new individual by parent crossover [returns individual]"""
        number_items = self._problem.get_number_items()

        length = np.random.randint(1, number_items)
        start = np.random.randint(1, number_items - length + 1)
        end = start + length

        part_a = parents[0][start:end]
        part_b = list(filter(lambda x: x not in part_a, parents[1]))

        child = part_b[:start] + part_a + part_b[start:]

        return child

    def mutate(self, individual):
        """Mutate individual"""
        number_items = self._problem.get_number_items()

        if (np.random.random() <= self._param_mutation_probability):
            index_i = np.random.randint(1, number_items)
            index_j = np.random.randint(1, number_items)
            if (index_i != index_j):
                value_i = individual[index_i]  # swap index i with j
                individual[index_i] = individual[index_j]
                individual[index_j] = value_i

    def mutate_alternative(self, individual):
        """Mutate individual"""
        for i in range(1, len(individual) - 2):
            if (np.random.random() <= self._param_mutation_probability):
                value_i = individual[i]  # swap index i with i + 1
                individual[i] = individual[i+1]
                individual[i+1] = value_i

    def create_random_individual(self):
        """Create random individual [returns individual]"""
        number_items = self._problem.get_number_items()

        new_individual = list(range(1, number_items))
        np.random.shuffle(new_individual)
        new_individual.insert(0, 0)
        new_individual.append(0)

        return new_individual
