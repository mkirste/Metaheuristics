import numpy as np
from metaheuristics.simulatedannealing import SimulatedAnnealing


class SimulatedAnnealingForTravelingSalesman(SimulatedAnnealing):
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

        solution = list(range(1, number_items))
        solution.insert(0, 0)
        solution.append(0)

        return solution

    def create_random_neighbor(self, solution):
        """Create random neighbor [returns solution]"""
        ############
        number_items = self._problem.get_number_items()
        index_i = np.random.randint(1, number_items)
        index_j = np.random.randint(1, number_items)

        neighbor = solution.copy()
        if (index_i != index_j):
            value_i = neighbor[index_i]  # swap index i with j
            neighbor[index_i] = neighbor[index_j]
            neighbor[index_j] = value_i

        return neighbor
