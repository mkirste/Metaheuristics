import numpy as np
from metaheuristics.tabusearch import TabuSearch


class TabuSearchForKnapsack(TabuSearch):
    def __init__(
        self,
        knapsack_problem,
        param_max_size_tabulist=10,
        param_max_number_iterations=100,
        param_logging=False
    ):
        super().__init__(param_max_size_tabulist, param_max_number_iterations, param_logging)
        self._knapsack_problem = knapsack_problem

    def get_solution_representation(self, candidate):
        """Get solution representation [returns solution representation]"""
        return np.array2string(candidate.solution, precision=0, separator=',')

    def create_start_solution(self):
        """Create start solution [returns solution]"""
        number_items = self._knapsack_problem.get_number_items()
        return np.zeros(number_items)

    def create_neighborhood(self, solution):
        """Create neighborhood for current solution [returns list of solutions]"""
        neighborhood = []

        for i in range(len(solution)):
            neighbor = np.copy(solution)
            neighbor[i] = 1 - neighbor[i]  # flip value
            neighborhood.append(neighbor)

        return neighborhood

    def fitness(self, solution):
        """Calculate fitness of solution (higher fitness indicates better solution) [returns fitness]"""
        self._knapsack_problem.set_solution(solution)
        return self._knapsack_problem.calculate_fitness()
