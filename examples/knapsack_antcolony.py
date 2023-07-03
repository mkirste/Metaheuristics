import numpy as np
import random
from metaheuristics.antcolony import AntColony


class AntColonyForKnapsack(AntColony):
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
        self._pheromone_trails = np.zeros(number_items)  # pheromone trail for each item

    def initialize(self):
        """Start Variables"""
        AntColony.initialize(self)
        number_items = self._problem.get_number_items()
        self._pheromone_trails = np.full(number_items, 0.1)

    def create_ant_solution(self):
        """Create ant solution [returns solution as candidate tuple]"""
        number_items = self._problem.get_number_items()
        data = self._problem.get_data()

        item_attractiveness = data["values"] / data["weights"]
        selected_items = []
        remaining_items = [x for x in range(number_items)]
        bag_weight = 0

        while len(remaining_items) > 0:
            remaining_items_selection = [
                item_attractiveness[x] * self._pheromone_trails[x]
                for x in remaining_items
            ]
            remaining_items_probabilities = remaining_items_selection / sum(
                remaining_items_selection
            )

            new_item = random.choices(remaining_items, remaining_items_probabilities)[0]

            selected_items.append(new_item)
            bag_weight += data["weights"][new_item]

            remaining_items.remove(new_item)
            remaining_items = list(
                filter(
                    lambda x: data["weights"][x] <= data["bag_size"] - bag_weight,
                    [x for x in remaining_items],
                )
            )

        ant_solution = np.array(
            [i in selected_items for i in range(number_items)]
        ).astype(int)

        """
        # Alternative using numpy
        item_attractiveness = data["values"] / data["weights"]
        bag_weight = 0
        ant_solution = np.zeros(number_items)

        while True:
            item_candidates = 1 - ant_solution  # items that are not in the bag
            item_candidates = item_candidates * np.where(
                data["weights"] > data["bag_size"] - bag_weight, 0, 1
            )  # items that can be added to bag due to capacity constraint
            if np.sum(item_candidates) == 0:
                break  # no remaining candidates

            item_selection = (
                item_attractiveness * self._pheromone_trails * item_candidates
            )
            item_probabilities = item_selection / np.sum(item_selection)

            new_item_index = np.random.choice(number_items, 1, p=item_probabilities)

            ant_solution[new_item_index] = 1
            bag_weight += data["weights"][new_item_index]
        """

        return ant_solution

    def update_pheromone_trails(self):
        """Update pheromone trails (evaporation mechanism and pheromone updates)"""
        number_items = self._problem.get_number_items()

        # evaporation mechanism
        self._pheromone_trails = self._pheromone_trails * self._param_evaporation_factor

        # pheromone updates
        pheromone_updates = np.zeros(number_items)
        for index, ant_solution in enumerate(self._ant_solutions):
            pheromone_updates = pheromone_updates + (
                ant_solution.solution * ant_solution.fitness
            )

        self._pheromone_trails = (
            self._pheromone_trails
            + (1 - self._param_evaporation_factor) * pheromone_updates
        )
