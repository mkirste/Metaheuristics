import numpy as np
from metaheuristics.hillclimbing import HillClimbing


class HillClimbingForKnapsack(HillClimbing):
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

    def create_neighborhood(self, solution):
        """Create neighborhood for current solution [returns list of solutions]"""
        neighborhood = []

        for i in range(len(solution)):
            neighbor = np.copy(solution)
            neighbor[i] = 1 - neighbor[i]  # flip value
            neighborhood.append(neighbor)

        return neighborhood
