from abc import ABC, abstractmethod
from collections import namedtuple
import random
from metaheuristics.algorithm import Algorithm

Log = namedtuple("Log", "best_fitness average_fitness")


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        problem,
        param_population_size,
        param_max_number_iterations,
        param_iteration_logging=False,
    ):
        super().__init__(problem, param_max_number_iterations, param_iteration_logging)
        self._param_population_size = param_population_size
        self._population = []  # list of individuals (i.e. solutions)

    # Algorithm
    def algorithm(self):
        """Algorithm [returns best solution]"""
        self.initialize()  # create start population

        while self.termination() == False:
            self.selection()  # select best individuals from population
            self.generation()  # create individuals for new generation
            self.update_iteration_counter()
            self.log_iteration()

        self._population = sorted(
            self._population, key=lambda x: -self.solution_fitness(x)
        )

        return self._population[0]

    def algorithm_name(self):
        return "GeneticAlgorithm"

    # Basic algorithm steps
    def initialize(self):
        """Initialize first population"""
        for i in range(self._param_population_size):
            self._population.append(self.create_random_individual())

    def selection(self):
        """Select part of population"""
        self._population = sorted(
            self._population, key=lambda x: -self.solution_fitness(x)
        )
        self._population = self._population[:self._param_population_size//2]

    def generation(self):
        """Create new generation"""
        new_population = []
        new_population = self._population[:self._param_population_size//10]

        while len(new_population) < self._param_population_size:
            parents = random.choices(self._population, k=2)
            child = self.crossover(parents)
            self.mutate(child)
            new_population.append(child)
        self._population = new_population

    def log_iteration(self):
        """Create log"""
        if self._param_iteration_logging:  # Case logging
            population_fitness = [self.solution_fitness(
                i) for i in self._population]
            best_fitness = max(population_fitness)
            average_fitness = sum(population_fitness) / len(population_fitness)
            log = Log(best_fitness, average_fitness)
            super().log_iteration(log)

    # Domain specific functions
    @abstractmethod
    def crossover(self, parents):
        """Create new individual by parent crossover [returns individual]"""
        return None

    @abstractmethod
    def mutate(self, individual):
        """Mutate individual"""
        pass

    @abstractmethod
    def create_random_individual(self):
        """Create random individual [returns individual]"""
        pass
