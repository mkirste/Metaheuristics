from abc import ABC, abstractmethod
from collections import namedtuple
from metaheuristics.algorithm import Algorithm

Candidate = namedtuple("Candidate", "solution fitness")
Log = namedtuple("Log", "best_fitness candidate_fitness")


class TabuSearch(Algorithm):
    def __init__(
        self,
        param_max_size_tabulist,
        param_max_number_iterations,
        param_iteration_logging=False,
    ):
        super().__init__(param_max_number_iterations, param_iteration_logging)
        self._param_max_size_tabulist = param_max_size_tabulist
        self._tabulist = []
        self._solution = None  # Candidate tuple (solution, fitness)

    # Algorithm
    def algorithm(self):
        """Algorithm [returns best solution]"""
        self.initialize()  # create start soluation

        candidate = self._solution  # Candidate tuple (solution, fitness)
        while self.termination() == False:
            candidates = self.create_candidates(
                candidate)  # create candidate list
            candidate = self.select_candidate(
                candidate, candidates)  # select best candidate
            self.update_tabulist(candidate)  # update tabu list
            self.update_solution(candidate)  # detect new best solution
            self.update_iteration_counter()
            self.log_iteration(candidate)

        return self._solution.solution

    # Basic algorithm steps
    def initialize(self):
        """Initialize start solution"""
        start_solution = self.create_start_solution()
        start_fitness = self.fitness(start_solution)
        self._solution = Candidate(start_solution, start_fitness)

    def create_candidates(self, candidate):
        """Create candidates (returns list (solution, fitness) tuple)"""
        return [Candidate(s, None) for s in self.create_neighborhood(candidate.solution)]

    def select_candidate(self, candidate, candidates):
        """Select best candidate"""
        best_candidate = None
        for candidate in candidates:
            if (self.get_solution_representation(candidate) not in self._tabulist):
                if (candidate.fitness == None):  # calculate fitness
                    candidate = Candidate(
                        candidate.solution, self.fitness(candidate.solution))
                if (best_candidate == None or candidate.fitness > best_candidate.fitness):
                    best_candidate = candidate

        if (best_candidate != None):
            return best_candidate
        else:
            return candidate

    def update_tabulist(self, candidate):
        """Update candidate list"""
        self._tabulist.append(self.get_solution_representation(candidate))
        if (len(self._tabulist) > self._param_max_size_tabulist):
            self._tabulist.pop(0)

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
    def get_solution_representation(self, candidate):
        """Get solution representation [returns solution representation]"""
        pass  # may be an hash value of the candidate.solution

    @abstractmethod
    def create_start_solution(self):
        """Create start solution [returns solution]"""
        pass

    @abstractmethod
    def create_neighborhood(self, solution):
        """Create neighborhood for current solution [returns list of solutions]"""
        return None
