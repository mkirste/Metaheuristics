import numpy as np
from problems.problem import Problem

# data["distances"]    [cities * cities]           (nd.array 2dim float)
# solution             (city indices ...)          (list)


class TravelingSalesmanProblem(Problem):
    def __init__(self, data, solution):
        super().__init__(data, solution)

    @classmethod
    def create_new_random(cls, number_items=100):
        data = {}

        data["distances"] = np.random.rand(number_items, number_items) * 100
        data["distances"] = np.around(data["distances"], decimals=0)
        # [i,i] = 0
        np.fill_diagonal(data["distances"], 0)
        # [i,j] = [j,i]
        data["distances"] = data["distances"] * \
            np.tri(number_items, number_items, -1)
        data["distances"] = data["distances"] + np.transpose(data["distances"])

        solution = []
        solution = list(range(1, number_items))
        solution.insert(0, 0)
        solution.append(0)

        return TravelingSalesmanProblem(data, solution)

    # Getters
    def get_number_items(self):
        """Get number of items"""
        return self._data["distances"][0].size

    # Functions (general) [implementations]
    def check_data_solution(self, data, solution):
        if data["distances"].shape[0] != data["distances"].shape[1]:
            raise ValueError(
                "Length of first dimension of distances must be same as second dimension!")

    def check_new_solution(self, solution):
        invalid_indices = list(
            filter(lambda x: x > self.get_number_items() - 1, solution))
        return len(invalid_indices) == 0

    def calculate_fitness(self):
        """Calculate fitness"""
        if (self.is_feasible()):
            return -self.calculate_traveldistance()
        else:
            return -np.inf

    def is_feasible(self):
        """Check if solution is feasible"""
        return self.check_restrictions()

   # Functions (problem specific)
    def calculate_traveldistance(self):
        """Calculate travel distance"""
        costs = 0
        for i in range(len(self._solution) - 1):
            city_from = self._solution[i]
            city_to = self._solution[i + 1]
            costs += self._data["distances"][city_from][city_to]
        return costs

    def check_restrictions(self):
        if (len(self._solution) - 1 != self.get_number_items()):
            return False  # number of visits

        if (self._solution[0] != 0 or self._solution[-1]):
            return False  # start and end in city 0

        required_cities = set(range(1, self.get_number_items()))
        visited_cities = set(self._solution[1:-1])
        if (len(required_cities - visited_cities) != 0):
            return False  # all cities visited

        return True


"""
Alternative with solution as 2dim nd array
    def calculate_costs(self):
        return np.sum(self._data_items * self._solution)

    def check_restrictions(self):
        if (np.all(np.sum(self._solution, axis=0) == 1) == False):
            return False  # all cities visited once

        if (np.all(np.sum(self._solution, axis=1) == 1) == False):
            return False  # all cities left once

        city = 0
        for i in range(self._solution.shape[0] - 1):
            city = np.nonzero(self._solution[city])[0][0]
            if (city == 0):
                return False  # subtour

        return True
"""
