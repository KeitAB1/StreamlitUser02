import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import logging
from utils import save_convergence_history, save_performance_metrics
from optimization_utils import evaluate_parallel, evaluate_with_cache
from optimization_utils import apply_adaptive_sa

class SA_with_Batch:
    def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, lambda_1, lambda_2,
                 lambda_3, lambda_4, num_positions, num_plates, dataset_name, objectives, use_adaptive):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.num_positions = num_positions
        self.num_plates = num_plates
        self.dataset_name = dataset_name
        self.objectives = objectives
        self.use_adaptive = use_adaptive

        self.best_position = None
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.convergence_data = []
        self.temperature_data = []
        self.adaptive_param_data = []
        self.start_time = None
        self.cache = {}
        self.score_changes = []

    def evaluate_with_cache(self, position):
        return evaluate_with_cache(self.cache, position, self.evaluate)

    def evaluate(self, position):
        try:
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(position)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(position)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(position)
            space_utilization = self.objectives.maximize_space_utilization_v3(position)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)
            return score
        except Exception as e:
            logging.error(f"Error in evaluation: {e}")
            return np.inf

    def optimize(self):
        initial_position = np.random.randint(0, self.num_positions, size=self.num_plates)
        return self.optimize_from_position(initial_position)

    def optimize_from_position(self, initial_position):
        current_temperature = self.initial_temperature
        current_position = initial_position
        current_score = self.evaluate_with_cache(current_position)

        self.best_position = current_position.copy()
        self.best_score = current_score
        self.worst_score = current_score
        self.start_time = time.time()

        scores = []
        unsuccessful_attempts = 0

        progress_bar = st.progress(0)

        with st.spinner("正在运行模拟退火优化..."):
            for iteration in range(self.max_iterations):
                if current_temperature < self.min_temperature:
                    break

                new_positions = [current_position.copy() for _ in range(5)]
                for new_position in new_positions:
                    random_index = np.random.randint(0, len(current_position))
                    new_position[random_index] = np.random.randint(0, self.num_positions)

                new_scores = evaluate_parallel(new_positions, self.evaluate_with_cache)

                best_new_score = min(new_scores)
                best_new_position = new_positions[new_scores.index(best_new_score)]

                delta = best_new_score - current_score
                if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                    current_position = best_new_position
                    current_score = best_new_score
                else:
                    unsuccessful_attempts += 1

                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = current_position.copy()

                if current_score > self.worst_score:
                    self.worst_score = current_score

                scores.append(current_score)
                self.score_changes.append(delta)

                current_temperature, self.cooling_rate = apply_adaptive_sa(
                    current_temperature, self.cooling_rate, delta, self.use_adaptive)

                if self.use_adaptive:
                    self.record_adaptive_params()

                # 记录收敛数据而不绘图
                self.convergence_data.append([iteration + 1, self.best_score])
                self.temperature_data.append(current_temperature)

                # 更新进度条
                progress_percentage = (iteration + 1) / self.max_iterations
                progress_bar.progress(progress_percentage)

        # 优化结束后清除进度条
        progress_bar.empty()

        avg_score = np.mean(scores)
        score_std = np.std(scores)
        time_elapsed = time.time() - self.start_time

        self.save_metrics(time_elapsed, avg_score, score_std, unsuccessful_attempts)

        # 保存收敛历史数据
        history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "SA")
        save_convergence_history(self.convergence_data, "SA", self.dataset_name, history_data_dir)

        return self.best_position, self.best_score

    def record_adaptive_params(self):
        self.adaptive_param_data.append({'cooling_rate': self.cooling_rate})

    def save_metrics(self, time_elapsed, avg_score, score_std, unsuccessful_attempts):
        iterations = len(self.convergence_data)
        total_improvement = np.sum(self.score_changes)
        self.worst_score = max(self.score_changes)

        # 确保避免空的 convergence_data 导致的索引错误
        if iterations > 1:
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
        else:
            convergence_rate_value = 0

        save_performance_metrics(
            self.best_score,
            self.worst_score,
            total_improvement,
            convergence_rate_value,
            iterations,
            time_elapsed,
            self.convergence_data,
            len(self.adaptive_param_data),
            self.dataset_name,
            "SA"
        )