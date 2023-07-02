import numpy as np
from problems.problem import Problem


# data["values"]     [items] (nd.array 1dim float)
# data["weights"]    [items] (nd.array 1dim float)
# data["bag_size"]           (float)
# solution           [items] (nd.array 1dim integer 0,1)


class KnapsackProblem(Problem):
    def __init__(self, data, solution):
        super().__init__(data, solution)

    @classmethod
    def create_new_random(cls, number_items=100, average_number_items_bag=10):
        data = {}

        data["values"] = np.random.rand(number_items) * 100
        data["values"] = np.around(data["values"], decimals=0)

        data["weights"] = np.random.rand(number_items) * 100
        data["weights"] = np.around(data["weights"], decimals=0)
        data["weights"][data["weights"] == 0] = 1

        data["bag_size"] = average_number_items_bag * 50

        solution = np.zeros(number_items)

        return KnapsackProblem(data, solution)

    # Getters
    def get_number_items(self):
        """Get number of items"""
        return self._data["values"].size

    # Functions (general) [implementations]
    def check_data_solution(self, data, solution):
        if data["values"].ndim != 1:
            raise ValueError("Values must be an array with one dimension!")
        if data["weights"].ndim != 1:
            raise ValueError("Weights must be an array with one dimension!")
        if data["values"].size != data["weights"].size:
            raise ValueError("Length of values and weights must be same!")
        if data["values"].size != solution.size:
            raise ValueError("Length of values must be the same as length of solution!")
        return True

    def check_new_solution(self, solution):
        if self._solution.size != solution.size:
            raise ValueError(
                "New solution must contain excatly same number of items ({})!".format(
                    self._solution.size
                )
            )
        return True

    def calculate_fitness(self):
        """Calculate fitness"""
        weight = self.calculate_weight()
        fitness = self.calculate_value()
        if self.check_weight_restriction(weight):
            return fitness
        else:
            return -weight

    def is_feasible(self):
        """Check if solution is feasible"""
        weight = self.calculate_weight()
        return self.check_weight_restriction(weight)

    # Functions (problem specific)
    def calculate_value(self):
        """Calculate value of solution"""
        return self._data["values"].dot(self._solution)

    def calculate_weight(self):
        """Calculate weight of solution"""
        return self._solution.dot(self._data["weights"])

    def check_weight_restriction(self, weight):
        """Calculate weight restriction of solution"""
        return weight <= self._data["bag_size"]
