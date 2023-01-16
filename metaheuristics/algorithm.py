from abc import abstractmethod
import time


class Algorithm:
    def __init__(
        self,
        problem,
        param_max_number_iterations=100,
        param_iteration_logging=False,
    ):
        self._problem = problem
        self._best_solution = None

        self._param_max_number_iterations = param_max_number_iterations
        self._param_iteration_logging = param_iteration_logging

        self._iteration_counter = 0
        self._iteration_log = []  # list of logs for each interation

        self._runtime = None  # runtime in seconds

    def start(self):
        """Start algorithm and return best solution, fitness """
        start_time = time.process_time()

        best_solution = self.algorithm()  # start algorithm
        best_fitness = self.solution_fitness(best_solution)

        self._best_solution = {
            "solution": best_solution, "fitness": best_fitness}

        self._runtime = time.process_time() - start_time

        if (self._param_iteration_logging == True):
            print("Fitness: {}; Runtime: {} seconds".format(
                best_fitness, self._runtime))

        return best_solution, best_fitness

    # Getters
    def get_problem(self):
        """ Returns problem """
        return self._problem

    def get_best_solution(self):
        """ Returns best solution [Dictionary (solution, fitness)] """
        return self._best_solution

    def get_log(self):
        return self._iteration_log

    def get_runtime(self):
        """Runtime in seconds"""
        return self._runtime

    # Domain specific functions
    @abstractmethod
    def algorithm(self):
        """Algorithm [returns best solution]"""
        pass

    @abstractmethod
    def algorithm_name(self):
        """Algorithm Name [returns string]"""
        pass

    # Further functions
    def solution_fitness(self, solution):
        """Calculate fitness of solution for the problem [returns fitness]"""
        self._problem.set_solution(solution)
        return self._problem.calculate_fitness()

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
