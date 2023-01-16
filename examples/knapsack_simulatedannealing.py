import numpy as np
from metaheuristics.simulatedannealing import SimulatedAnnealing


class SimulatedAnnealingForKnapsack(SimulatedAnnealing):
    def __init__(
        self,
        problem,
        param_max_number_iterations=100,
        param_logging=False
    ):
        super().__init__(problem, param_max_number_iterations, param_logging)

    def create_start_solution(self):
        """Create start solution [returns solution]"""
        number_items = self._problem.get_number_items()
        return np.zeros(number_items)

    def create_random_neighbor(self, solution):
        """Create random neighbor [returns solution]"""
        number_items = self._problem.get_number_items()
        index = np.random.randint(0, number_items)

        neighbor = np.copy(solution)
        neighbor[index] = 1 - neighbor[index]  # flip value

        return neighbor
