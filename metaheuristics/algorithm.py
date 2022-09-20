from abc import ABC, abstractmethod
import time


class Algorithm:
    def __init__(
        self,
        param_max_number_iterations=100,
        param_iteration_logging=False,
    ):
        self._param_max_number_iterations = param_max_number_iterations
        self._param_iteration_logging = param_iteration_logging
        self._iteration_counter = 0
        self._iteration_log = []  # list of logs for each interation
        self._runtime = None  # runtime in seconds

    def start(self):
        """Start algorithm and return best solution"""
        start_time = time.process_time()
        best_solution = self.algorithm()  # start algorithm
        self._runtime = time.process_time() - start_time
        if (self._param_iteration_logging == True):
            print("Fitness: {}; Runtime: {} seconds".format(
                self.fitness(best_solution), self._runtime))
        return best_solution

    def termination(self):
        """Check stopp condition"""
        return self._iteration_counter >= self._param_max_number_iterations

    def update_iteration_counter(self):
        self._iteration_counter += 1

    def log_iteration(self, iteration_info):
        """Create iteration log"""
        if self._param_iteration_logging:  # Case logging
            print("Iteration {}:".format(self._iteration_counter), iteration_info)
            self._iteration_log.append(iteration_info)

    # Domain specific functions
    @abstractmethod
    def algorithm(self):
        """Algorithm [returns best solution]"""
        pass

    @abstractmethod
    def fitness(self, solution):
        """Calculate fitness of solution (higher fitness indicates better solution) [returns float]"""
        pass

    # Further functions
    def get_log(self):
        return self._iteration_log

    def get_runtime(self):
        return self._runtime
