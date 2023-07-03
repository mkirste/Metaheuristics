from abc import ABC, abstractmethod
from collections import namedtuple
import random
from metaheuristics.algorithm import Algorithm

Candidate = namedtuple("Candidate", "solution fitness")
Log = namedtuple("Log", "best_fitness average_fitness")


class AntColony(Algorithm):
    def __init__(
        self,
        problem,
        param_evaporation_factor,
        param_number_ants,
        param_max_number_iterations,
        param_iteration_logging=False,
    ):
        super().__init__(problem, param_max_number_iterations, param_iteration_logging)
        self._param_evaporation_factor = param_evaporation_factor  # 0.95
        self._param_number_ants = param_number_ants
        self._ant_solutions = []  # list of ant solutions (solution, fitness)
        self._solution = None  # Candidate tuple (solution, fitness)

    # Algorithm
    def algorithm(self):
        """Algorithm [returns best solution]"""
        self.initialize()

        while self.termination() == False:
            self.generate_ant_solutions()  # solution generation by ants
            self.analize_ant_solutions()
            self.update_pheromone_trails()  # evaporation mechanism and pheromone updates
            self.update_iteration_counter()
            self.log_iteration()

        return self._solution.solution

    def algorithm_name(self):
        return "AntColonyOptimization"

    # Basic algorithm steps
    def initialize(self):
        """Start Variables"""
        self._ant_solutions = []
        self._solution = Candidate(
            self._problem.get_solution(), self._problem.calculate_fitness()
        )

    def generate_ant_solutions(self):
        """Generate solutions using ants"""
        self._ant_solutions = []
        for i in range(self._param_number_ants):
            new_ant_solution = self.create_ant_solution()
            new_ant_fitness = self.solution_fitness(new_ant_solution)
            new_ant_candidate = Candidate(new_ant_solution, new_ant_fitness)
            self._ant_solutions.append(new_ant_candidate)

    def analize_ant_solutions(self):
        for index, ant_solution in enumerate(self._ant_solutions):
            self.update_solution(ant_solution)

    def update_solution(self, candidate):
        """Update best solution [returns True if new best solution, otherwise False]"""
        if candidate.fitness > self._solution.fitness:
            self._solution = candidate
            return True
        else:
            return False

    def log_iteration(self):
        """Create log"""
        if self._param_iteration_logging:  # Case logging
            ant_fitness = [self.solution_fitness(i) for i in self._ant_solutions]
            best_fitness = max(ant_fitness)
            average_fitness = sum(ant_fitness) / len(ant_fitness)
            log = Log(best_fitness, average_fitness)
            super().log_iteration(log)

    # Domain specific functions
    @abstractmethod
    def create_ant_solution(self):
        """Create ant solution [returns solution]"""
        pass

    @abstractmethod
    def update_pheromone_trails(self):
        """Update pheromone trails (evaporation mechanism and pheromone updates)"""
        pass
