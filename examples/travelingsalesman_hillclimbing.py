import numpy as np
from metaheuristics.hillclimbing import HillClimbing


class HillClimbingForTravelingSalesman(HillClimbing):
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

    def create_neighborhood(self, solution):
        """Create neighborhood for current solution [returns list of solutions]"""
        neighborhood = []

        for i in range(1, len(solution) - 2):
            neighbor = solution.copy()
            value_i = neighbor[i]
            neighbor[i] = neighbor[i+1]  # swap index i with i + 1
            neighbor[i+1] = value_i
            neighborhood.append(neighbor)

        return neighborhood
