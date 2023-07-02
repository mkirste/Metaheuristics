import numpy as np
from metaheuristics.geneticalgorithm import GeneticAlgorithm


class GeneticAlgorithmForKnapsack(GeneticAlgorithm):
    def __init__(
        self,
        problem,
        param_mutation_probability=0.02,
        param_population_size=100,
        param_max_number_iterations=10,
        param_logging=False,
    ):
        super().__init__(
            problem, param_population_size, param_max_number_iterations, param_logging
        )
        self._param_mutation_probability = param_mutation_probability

    def crossover(self, parents):
        """Create new individual by parent crossover [returns individual]"""
        number_items = self._problem.get_number_items()
        return np.concatenate(
            (parents[0][: number_items // 2], parents[1][number_items // 2 :])
        )

    def mutate(self, individual):
        """Mutate individual"""
        for i, item in enumerate(individual):
            if np.random.random() <= self._param_mutation_probability:
                individual[i] = 1 - individual[i]  # flip value

    def create_random_individual(self):
        """Create random individual [returns individual]"""
        number_items = self._problem.get_number_items()
        data = self._problem.get_data()

        average_numer_items = data["bag_size"] / (sum(data["weights"]) / number_items)

        new_individual = np.random.rand(number_items)
        new_individual = new_individual - 0.5 + (average_numer_items / number_items)
        new_individual = np.around(new_individual)

        return new_individual
