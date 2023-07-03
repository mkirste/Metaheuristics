import numpy as np
import random
from metaheuristics.antcolony import AntColony


class AntColonyForTravelingSalesman(AntColony):
    def __init__(
        self,
        problem,
        param_evaporation_factor=0.95,
        param_number_ants=100,
        param_max_number_iterations=10,
        param_logging=False,
    ):
        super().__init__(
            problem,
            param_evaporation_factor,
            param_number_ants,
            param_max_number_iterations,
            param_logging,
        )
        number_items = problem.get_number_items()
        self._pheromone_trails = np.zeros(
            (number_items, number_items)
        )  # pheromone trail for each item

    def initialize(self):
        """Start Variables"""
        AntColony.initialize(self)
        number_items = self._problem.get_number_items()
        self._pheromone_trails = np.full((number_items, number_items), 0.1)

    def create_ant_solution(self):
        """Create ant solution [returns solution as candidate tuple]"""
        number_items = self._problem.get_number_items()
        data = self._problem.get_data()

        # edge_attractiveness = 1 / data["distances"]
        edge_attractiveness = np.divide(
            1, data["distances"], where=data["distances"] != 0
        )

        ant_solution = [0]
        remaining_cities = [x for x in range(number_items) if x > 0]

        while len(remaining_cities) > 0:
            city_from = ant_solution[-1]

            remaining_cities_selection = [
                edge_attractiveness[city_from][x] * self._pheromone_trails[city_from][x]
                for x in remaining_cities
            ]

            print("----------------------------------------------------------")
            print("data distances", data["distances"])
            print("remaining_cities", remaining_cities)
            print("edge_attractiveness", edge_attractiveness)
            print(
                "self._pheromone_trails,",
                self._pheromone_trails,
            )
            print("remaining_cities_selection", remaining_cities_selection)
            print(
                "remaining_cities_probabilities",
                remaining_cities_selection / sum(remaining_cities_selection),
            )
            print("")

            remaining_cities_probabilities = remaining_cities_selection / sum(
                remaining_cities_selection
            )

            city_next = random.choices(
                remaining_cities, remaining_cities_probabilities
            )[0]

            remaining_cities.remove(city_next)
            ant_solution.append(city_next)

        ant_solution.append(0)

        return ant_solution

    def update_pheromone_trails(self):
        """Update pheromone trails (evaporation mechanism and pheromone updates)"""
        number_items = self._problem.get_number_items()

        # evaporation mechanism
        self._pheromone_trails = self._pheromone_trails * self._param_evaporation_factor

        # pheromone updates
        pheromone_updates = np.zeros((number_items, number_items))
        for index, ant_solution in enumerate(self._ant_solutions):
            for i in range(len(ant_solution.solution) - 1):
                city_from = ant_solution.solution[i]
                city_to = ant_solution.solution[i + 1]
                pheromone_updates[city_from][city_to] += ant_solution.fitness

        self._pheromone_trails = (
            self._pheromone_trails
            + (1 - self._param_evaporation_factor) * pheromone_updates
        )
