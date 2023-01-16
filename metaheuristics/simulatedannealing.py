from abc import ABC, abstractmethod
from collections import namedtuple
import math
import numpy as np
from metaheuristics.algorithm import Algorithm

Candidate = namedtuple("Candidate", "solution fitness")
Log = namedtuple("Log", "best_fitness candidate_fitness")


class SimulatedAnnealing(Algorithm):
    def __init__(
        self,
        problem,
        param_max_number_iterations,
        param_iteration_logging=False,
    ):
        super().__init__(problem, param_max_number_iterations, param_iteration_logging)
        self._temperature = 1
        self._solution = None  # Candidate tuple (solution, fitness)

    # Algorithm
    def algorithm(self):
        """Algorithm [returns best solution]"""
        self.initialize()  # create start soluation

        candidate = self._solution  # Candidate tuple (solution, fitness)
        while self.termination() == False:
            candidate = self.select_candidate(candidate)  # select candidate
            self.update_temperature()  # update temperature
            self.update_solution(candidate)  # detect new best solution
            self.update_iteration_counter()
            self.log_iteration(candidate)

        return self._solution.solution

    def algorithm_name(self):
        return "SimulatedAnnealing"

    # Basic algorithm steps
    def initialize(self):
        """Initialize start solution"""
        start_solution = self.create_start_solution()
        start_fitness = self.solution_fitness(start_solution)
        self._solution = Candidate(start_solution, start_fitness)

    def select_candidate(self, candidate):
        """Select candidate"""
        new_solution = self.create_random_neighbor(candidate.solution)
        new_fitness = self.solution_fitness(new_solution)
        new_candidate = Candidate(new_solution, new_fitness)

        if (new_candidate.fitness > candidate.fitness):
            return new_candidate
        else:
            if (candidate.fitness != 0):
                probability = math.exp(
                    (new_candidate.fitness - candidate.fitness) / abs(candidate.fitness) / self._temperature)
                if (np.random.random() <= probability):
                    return new_candidate
        return candidate

    def update_temperature(self):
        end_temperature = 0.001  # start_temperature is 1
        proportion_iterations = (
            self._iteration_counter + 1) / self._param_max_number_iterations
        self._temperature = 1 - (1 - end_temperature) * \
            (proportion_iterations)  # linear
        # self._temperature = 1 * end_temperature**(proportion_iterations) # multiplicative

    def update_solution(self, candidate):
        """Update best solution [returns True if new best solution, otherweise False]"""
        if (candidate.fitness > self._solution.fitness):
            self._solution = candidate
            return True
        else:
            return False

    def log_iteration(self, candidate):
        """Create iteration log"""
        log = Log(self._solution.fitness, candidate.fitness)
        super().log_iteration(log)

    # Domain specific functions
    @abstractmethod
    def create_start_solution(self):
        """Create start solution [returns solution]"""
        pass

    @abstractmethod
    def create_random_neighbor(self, solution):
        """Create random neighbor [returns solution]"""
        pass
