import numpy as np
from problems.knapsack import KnapsackProblem
from examples.knapsack_hillclimbing import HillClimbingForKnapsack
from examples.knapsack_tabusearch import TabuSearchForKnapsack
from examples.knapsack_simulatedannealing import SimulatedAnnealingForKnapsack
from examples.knapsack_geneticalgorithm import GeneticAlgorithmForKnapsack
from problems.travelingsalesman import TravelingSalesmanProblem
from examples.travelingsalesman_hillclimbing import HillClimbingForTravelingSalesman
from examples.travelingsalesman_tabusearch import TabuSearchForTravelingSalesman
from examples.travelingsalesman_simulatedannealing import SimulatedAnnealingForTravelingSalesman
from examples.travelingsalesman_geneticalgorithm import GeneticAlgorithmForTravelingSalesman


def print_result(title, start_fitness, results):
    """Print result"""
    header_format = "|{:^20}|{:^14}|{:^13}|{:^15}|"
    row_format = "| {:<18} | {:>12} | {:>11.4f} | {:>13.6f} |"
    divider_format = "|{:-^20}|{:-^14}|{:-^13}|{:-^15}|"

    print("|{:-^65}|".format(""))
    print("|{:^65}|".format(title))
    print(divider_format.format("", "", "", ""))
    print(header_format.format(
        "Algorithm", "best fitness",  "improvement", "runtime (sec)"))
    print(divider_format.format("", "", "", ""))

    for result in results:
        if result.get_best_solution() != None:
            fitness = result.get_best_solution()["fitness"]
            improvement = fitness if start_fitness == 0 else abs(
                fitness / start_fitness - 1)
            print(row_format.format(result.algorithm_name(),
                                    fitness, improvement, result.get_runtime()))

    print(divider_format.format("", "", "", ""))
    print("")


"""
# KnapsackProblem
problem = KnapsackProblem.create_new_random(10, 4)
problem.set_solution(np.around(np.random.rand(10)))
problem.set_solution(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
print("data:", problem.get_data())
print("solution:", problem.get_solution())
print("value:", problem.calculate_value())
print("weight:", problem.calculate_weight())
"""

# KnapsackProblem Optimization
problem = KnapsackProblem.create_new_random(25, 5)  # 100, 25
start_fitness = problem.calculate_fitness()
logging = False

hc = HillClimbingForKnapsack(problem, 100, logging)
ts = TabuSearchForKnapsack(problem, 10, 100, logging)
sa = SimulatedAnnealingForKnapsack(problem, 10000, logging)
ga = GeneticAlgorithmForKnapsack(problem, 0.02, 50, 100, logging)
algorithms = [hc, ts, sa, ga]
for algorithm in algorithms:
    algorithm.start()
    pass

print_result("Knapsack Optimization", start_fitness, algorithms)


"""
# TravelingSalesmanProblem
problem = TravelingSalesmanProblem.create_new_random(4)
problem.set_solution([0, 1, 2, 3, 0])
problem.is_feasible()
print(problem.get_data())
print(problem.calculate_traveldistance())
"""

# TravelingSalesman Optimization
problem = TravelingSalesmanProblem.create_new_random(25)  # 100, 25
start_fitness = problem.calculate_fitness()
logging = False

hc = HillClimbingForTravelingSalesman(problem, 100, logging)
ts = TabuSearchForTravelingSalesman(problem, 10, 100, logging)
sa = SimulatedAnnealingForTravelingSalesman(problem, 10000, logging)
ga = GeneticAlgorithmForTravelingSalesman(problem, 0.02, 50, 100, logging)
algorithms = [hc, ts, sa, ga]
for algorithm in algorithms:
    algorithm.start()
    pass

print_result("Knapsack TravelingSalesman", start_fitness, algorithms)
