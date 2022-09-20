import numpy as np


class KnapsackProblem:
    def __init__(self, bag_size, items, solution):
        if items.ndim != 2:
            raise ValueError("Items must be an array with two dimensions!")
        if items.shape[0] != 2:
            raise ValueError("Length of first dimension of items must be 2!")
        if items.shape[1] != solution.size:
            raise ValueError(
                "Length of second dimension of items must be the same as length of solution!")
        self._data_bag_size = bag_size
        self._data_items = items  # [weights, values; items]
        self._solution = solution  # [items]

    @classmethod
    def create_new_random(cls, number_items=100, average_number_items_bag=10):
        bag_size = average_number_items_bag * 50  # average number items
        items = np.random.rand(2, number_items) * 100
        items = np.around(items, decimals=0)
        solution = np.zeros(number_items)
        return KnapsackProblem(bag_size, items, solution)

    # Getters
    def get_number_items(self):
        """Get number of items"""
        return self._data_items[1].size

    def get_data(self):
        """Get data"""
        return self._data_items, self._data_bag_size

    def get_solution(self):
        """Get solution"""
        return self._solution

    def set_solution(self, solution):
        """Set new solution"""
        if self._data_items.shape[1] != solution.size:
            raise ValueError("Solution must contain excatly same number as number of items ({})!".format(
                self._data_items.shape[1]))
        self._solution = solution

    # Functions
    def calculate_value(self):
        """Calculate value"""
        return self._data_items[1].dot(self._solution)

    def calculate_weight(self):
        """Calculate weight"""
        return self._solution.dot(self._data_items[0])

    def check_weight_restriction(self, weight):
        return weight <= self._data_bag_size

    def calculate_fitness(self):
        """Calculate fitness"""
        weight = self.calculate_weight()
        fitness = self.calculate_value()
        if (self.check_weight_restriction(weight)):
            return fitness
        else:
            return -weight

    def is_feasible(self):
        """Check if solution is feasible"""
        weight = self.calculate_weight()
        return self.check_weight_restriction(weight)
