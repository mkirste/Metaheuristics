import numpy as np
from problems.knapsack import KnapsackProblem
from examples.hillclimbing_knapsack import HillClimbingForKnapsack
from examples.tabusearch_knapsack import TabuSearchForKnapsack
from examples.simulatedannealing_knapsack import SimulatedAnnealingForKnapsack
from examples.geneticalgorithm_knapsack import GeneticAlgorithmForKnapsack

test_problem = KnapsackProblem.create_new_random(25, 5)  # 100, 25
logging = False
"""
test_problem.set_solution(np.around(np.random.rand(10)))
test_problem.set_solution(np.array([1,1,1,0,0,0,0,0,0,0]))
print("data:", test_problem.get_data())
print("solution:", test_problem.get_solution())
print("weight:", test_problem.calculate_weight())
print("value:", test_problem.calculate_value())
"""

# HillClimbing
hc = HillClimbingForKnapsack(test_problem, 100, logging)
hc_solution = hc.start()
test_problem.set_solution(hc_solution)
hc_fitness = test_problem.calculate_fitness()

# TabuSearch
ts = TabuSearchForKnapsack(test_problem, 10, 100, logging)
ts_solution = ts.start()
test_problem.set_solution(ts_solution)
ts_fitness = test_problem.calculate_fitness()

# SimulatedAnnealing
sa = SimulatedAnnealingForKnapsack(test_problem, 5000, logging)
sa_solution = sa.start()
test_problem.set_solution(sa_solution)
sa_fitness = test_problem.calculate_fitness()

# GeneticAlgorithm
ga = GeneticAlgorithmForKnapsack(test_problem, 0.02, 20, 100, logging)
ga_solution = ga.start()
test_problem.set_solution(ga_solution)
ga_fitness = test_problem.calculate_fitness()


print("Algorithm: best fitness (duration)")
print("HillClimbing: {} ({} sec)".format(hc_fitness, hc.get_runtime()))
print("TabuSearch: {} ({} sec)".format(ts_fitness, ts.get_runtime()))
print("SimulatedAnnealing: {} ({} sec)".format(sa_fitness, sa.get_runtime()))
print("GeneticAlgorithm: {} ({} sec)".format(ga_fitness, ga.get_runtime()))
