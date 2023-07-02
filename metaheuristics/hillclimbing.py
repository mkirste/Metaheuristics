from abc import ABC, abstractmethod
from collections import namedtuple
from metaheuristics.algorithm import Algorithm

Candidate = namedtuple("Candidate", "solution fitness")
Log = namedtuple("Log", "best_fitness candidate_fitness")


class HillClimbing(Algorithm):
    def __init__(
        self,
        problem,
        param_max_number_iterations,
        param_iteration_logging=False,
    ):
        super().__init__(problem, param_max_number_iterations, param_iteration_logging)
        self._solution = None  # Candidate tuple (solution, fitness)

    # Algorithm
    def algorithm(self):
        """Algorithm [returns best solution]"""
        self.initialize()  # create start solution

        candidate = self._solution  # Candidate tuple (solution, fitness)
        update = True
        while self.termination() == False and update == True:
            candidates = self.create_candidates(candidate)  # create candidate list
            candidate = self.select_candidate(
                candidate, candidates
            )  # select best candidate
            update = self.update_solution(candidate)  # detect new best solution
            self.update_iteration_counter()
            self.log_iteration(candidate)

        return self._solution.solution

    def algorithm_name(self):
        return "HillClimbing"

    # Basic algorithm steps
    def initialize(self):
        """Initialize start solution"""
        start_solution = self.create_start_solution()
        start_fitness = self.solution_fitness(start_solution)
        self._solution = Candidate(start_solution, start_fitness)

    def create_candidates(self, candidate):
        """Create candidates (returns list (solution, fitness) tuple)"""
        return [
            Candidate(s, None) for s in self.create_neighborhood(candidate.solution)
        ]

    def select_candidate(self, candidate, candidates):
        """Select best candidate"""
        best_candidate = None
        for candidate in candidates:
            if candidate.fitness == None:  # calculate fitness
                candidate = Candidate(
                    candidate.solution, self.solution_fitness(candidate.solution)
                )
            if best_candidate == None or candidate.fitness > best_candidate.fitness:
                best_candidate = candidate

        if best_candidate != None:
            return best_candidate
        else:
            return candidate

    def update_solution(self, candidate):
        """Update best solution [returns True if new best solution, otherwise False]"""
        if candidate.fitness > self._solution.fitness:
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
    def create_neighborhood(self, solution):
        """Create neighborhood for current solution [returns list of solutions]"""
        return None
