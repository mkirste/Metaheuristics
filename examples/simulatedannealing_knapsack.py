import numpy as np
from metaheuristics.simulatedannealing import SimulatedAnnealing


class SimulatedAnnealingForKnapsack(SimulatedAnnealing):
    def __init__(
        self,
        knapsack_problem,
        param_max_number_iterations=100,
        param_logging=False
    ):
        super().__init__(param_max_number_iterations, param_logging)
        self._knapsack_problem = knapsack_problem

    def create_start_solution(self):
        """Create start solution [returns solution]"""
        number_items = self._knapsack_problem.get_number_items()
        return np.zeros(number_items)

    def create_random_neighbor(self, solution):
        """Create start solution [returns solution]"""
        number_items = self._knapsack_problem.get_number_items()
        index = np.random.randint(0, number_items)

        neighbor = np.copy(solution)
        neighbor[index] = 1 - neighbor[index]  # flip value

        return neighbor

    def fitness(self, solution):
        """Calculate fitness of solution (higher fitness indicates better solution) [returns fitness]"""
        self._knapsack_problem.set_solution(solution)
        return self._knapsack_problem.calculate_fitness()
