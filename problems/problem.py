from abc import abstractmethod


class Problem:
    def __init__(self, data, solution):
        self.check_data_solution(data, solution)
        self._data = data
        self._solution = solution

    # Getters
    def get_data(self):
        """Get data"""
        return self._data

    def get_solution(self):
        """Get solution"""
        return self._solution

    def set_solution(self, solution):
        """Set new solution"""
        if (self.check_new_solution(solution)):
            self._solution = solution

    # Functions
    @abstractmethod
    def check_data_solution(self, data, solution):
        """Checks data and solution (throws error) [returns float] """
        pass

    @abstractmethod
    def check_new_solution(self, solution):
        """Checks new solution (throws error) [returns float] """
        pass

    @abstractmethod
    def calculate_fitness(self):
        """Calculate fitness of solution (higher fitness indicates better solution) [returns float]"""
        pass

    @abstractmethod
    def is_feasible(self):
        """Check if solution is feasible"""
        pass
