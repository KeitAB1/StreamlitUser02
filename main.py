# main.py
import numpy as np
import time

class SA_with_Batch:
    def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, lambda_1, lambda_2,
                 lambda_3, lambda_4, num_positions, dataset_name, objectives, use_adaptive=False):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.num_positions = num_positions
        self.dataset_name = dataset_name
        self.objectives = objectives
        self.use_adaptive = use_adaptive
        self.best_position = None
        self.best_score = np.inf
        self.convergence_data = []
        self.start_time = None

    def evaluate(self, position):
        combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(position)
        energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(position)
        balance_penalty = self.objectives.maximize_inventory_balance_v2(position)
        space_utilization = self.objectives.maximize_space_utilization_v3(position)

        score = (self.lambda_1 * combined_movement_turnover_penalty +
                 self.lambda_2 * energy_time_penalty +
                 self.lambda_3 * balance_penalty -
                 self.lambda_4 * space_utilization)
        return score

    def optimize(self):
        current_position = np.random.randint(0, self.num_positions, size=len(self.objectives.plates))
        current_score = self.evaluate(current_position)
        self.best_position = current_position.copy()
        self.best_score = current_score
        temperature = self.initial_temperature
        self.start_time = time.time()

        for iteration in range(self.max_iterations):
            if temperature < self.min_temperature:
                break

            new_position = current_position.copy()
            random_index = np.random.randint(0, len(new_position))
            new_position[random_index] = np.random.randint(0, self.num_positions)
            new_score = self.evaluate(new_position)

            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_position = new_position
                current_score = new_score

                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_position = current_position.copy()

            self.convergence_data.append([iteration + 1, self.best_score])
            temperature *= self.cooling_rate

        time_elapsed = time.time() - self.start_time
        print(f"Optimization completed in {time_elapsed} seconds.")
        return self.best_position, self.best_score
