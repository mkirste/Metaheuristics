import numpy as np
from problems.knapsack import KnapsackProblem
from examples.knapsack_hillclimbing import HillClimbingForKnapsack
from examples.knapsack_tabusearch import TabuSearchForKnapsack
from examples.knapsack_simulatedannealing import SimulatedAnnealingForKnapsack
from examples.knapsack_geneticalgorithm import GeneticAlgorithmForKnapsack
from examples.knapsack_antcolony import AntColonyForKnapsack
from problems.travelingsalesman import TravelingSalesmanProblem
from examples.travelingsalesman_hillclimbing import HillClimbingForTravelingSalesman
from examples.travelingsalesman_tabusearch import TabuSearchForTravelingSalesman
from examples.travelingsalesman_simulatedannealing import (
    SimulatedAnnealingForTravelingSalesman,
)
from examples.travelingsalesman_geneticalgorithm import (
    GeneticAlgorithmForTravelingSalesman,
)
from examples.travelingsalesman_antcolony import AntColonyForTravelingSalesman


def print_result(title, start_fitness, results):
    """Print result"""
    header_format = "|{:^24}|{:^14}|{:^13}|{:^12}|{:^15}|"
    row_format = "| {:<22} | {:>12.6f} | {:>11.4f} | {:>10.2%} | {:>13.6f} |"
    divider_format = "|{:-^24}|{:-^14}|{:-^13}|{:-^12}|{:-^15}|"

    print("|{:-^82}|".format(""))
    print("|{:^82}|".format(title))
    print(divider_format.format("", "", "", "", ""))
    print(
        header_format.format(
            "Algorithm", "best fitness", "improvement", "comparison", "runtime (sec)"
        )
    )
    print(divider_format.format("", "", "", "", ""))

    for result in results:
        if result.get_best_solution() != None:
            fitness = result.get_best_solution()["fitness"]
            improvement = (
                fitness if start_fitness == 0 else abs(fitness / start_fitness - 1)
            )
            comparison = fitness / max(
                [x.get_best_solution()["fitness"] for x in results]
            )
            print(
                row_format.format(
                    result.algorithm_name(),
                    fitness,
                    improvement,
                    comparison,
                    result.get_runtime(),
                )
            )

    print(divider_format.format("", "", "", "", ""))
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
ac = AntColonyForKnapsack(problem, 0.95, 12, 50, logging)
algorithms = [hc, ts, sa, ga, ac]
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
ac = AntColonyForTravelingSalesman(problem, 0.95, 10, 50, logging)
algorithms = [hc, ts, sa, ga, ac]
for algorithm in algorithms:
    algorithm.start()
    pass

print_result("Knapsack TravelingSalesman", start_fitness, algorithms)
