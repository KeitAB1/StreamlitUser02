import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils import save_convergence_history, save_performance_metrics
from optimization_utils import evaluate_parallel, evaluate_with_cache, run_distributed_optimization
from optimization_utils import apply_adaptive_pso, apply_adaptive_sa, apply_adaptive_ga, apply_adaptive_coea, apply_adaptive_eda
from concurrent.futures import ThreadPoolExecutor

class GA_with_Batch:
    cache = {}  # 缓存适应度结果，避免重复计算

    def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                 lambda_4, num_positions, dataset_name, objectives, plates, delivery_times, batches, use_adaptive):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.num_positions = num_positions
        self.population = [np.random.randint(0, num_positions, size=len(plates)) for _ in range(population_size)]
        self.best_position = None  # 修改为 best_position
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.best_improvement = np.inf
        self.prev_best_score = np.inf
        self.total_improvement = 0
        self.convergence_data = []
        self.stable_iterations = 0
        self.dataset_name = dataset_name
        self.start_time = None  # 用于记录优化过程的时间
        self.objectives = objectives  # 优化目标实例
        self.plates = plates
        self.delivery_times = delivery_times
        self.batches = batches
        self.heights = np.zeros(num_positions)
        self.use_adaptive = use_adaptive
        self.adaptive_param_data = []

    def fitness(self, individual):
        individual_tuple = tuple(individual)
        # 使用缓存机制避免重复计算
        return evaluate_with_cache(self.cache, individual_tuple, self._evaluate_fitness)

    def _evaluate_fitness(self, individual):
        # 计算适应度得分
        combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(individual)
        energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
        balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
        space_utilization = self.objectives.maximize_space_utilization_v3(individual)

        score = (self.lambda_1 * combined_movement_turnover_penalty +
                 self.lambda_2 * energy_time_penalty +
                 self.lambda_3 * balance_penalty -
                 self.lambda_4 * space_utilization)

        return score

    def select(self):
        """
        选择个体：使用并行计算评估适应度
        """
        fitness_scores = evaluate_parallel(self.population, self.fitness)
        fitness_scores = np.array(fitness_scores)
        probabilities = np.exp(-fitness_scores / np.sum(fitness_scores))
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, len(self.plates))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.randint(0, self.num_positions)
        return individual

    def optimize(self):
        # st.info("GA Optimization started...")  # 提供优化开始信息
        with st.spinner("Running GA Optimization..."):  # 提供运行时加载提示
            self.start_time = time.time()  # 记录开始时间

            progress_bar = st.progress(0)  # 初始化进度条

            for generation in range(self.generations):
                new_population = []
                selected_population = self.select()

                # 交叉和变异操作使用多线程并行处理
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for i in range(0, self.population_size, 2):
                        parent1 = selected_population[i]
                        parent2 = selected_population[min(i + 1, self.population_size - 1)]
                        futures.append(executor.submit(self.crossover, parent1, parent2))

                    for future in futures:
                        child1, child2 = future.result()
                        new_population.append(self.mutate(child1))
                        new_population.append(self.mutate(child2))

                self.population = new_population
                best_individual_gen = min(self.population, key=self.fitness)
                best_score_gen = self.fitness(best_individual_gen)

                self.worst_score = max(self.worst_score, best_score_gen)
                self.best_improvement = min(self.best_improvement, abs(self.prev_best_score - best_score_gen))
                self.total_improvement += abs(self.prev_best_score - best_score_gen)
                self.prev_best_score = best_score_gen

                if best_score_gen < self.best_score:
                    self.best_score = best_score_gen
                    self.best_position = best_individual_gen.copy()  # 这里使用 best_position
                    self.stable_iterations = 0  # 重置稳定迭代次数
                else:
                    self.stable_iterations += 1  # 计数稳定迭代次数

                # 记录并更新自适应参数
                if self.use_adaptive:
                    self.mutation_rate, self.crossover_rate = apply_adaptive_ga(
                        self.mutation_rate, self.crossover_rate, best_score_gen, self.best_score, self.use_adaptive
                    )
                    self.record_adaptive_params()

                # 记录收敛数据
                self.convergence_data.append([generation + 1, self.best_score])

                # 更新进度条
                progress_percentage = (generation + 1) / self.generations
                progress_bar.progress(progress_percentage)

                # print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

            # 优化结束后清除进度条
            progress_bar.empty()

            # 保存收敛历史数据
            time_elapsed = time.time() - self.start_time  # 计算总耗时
            iterations = len(self.convergence_data)
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            average_improvement = self.total_improvement / iterations if iterations > 0 else np.inf
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)

            metrics = {
                'Best Score': self.best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': self.best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': self.stable_iterations
            }

            self.save_metrics(metrics)

            history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "GA")
            save_convergence_history(self.convergence_data, "GA", self.dataset_name, history_data_dir)

    def record_adaptive_params(self):
        self.adaptive_param_data.append(
            {'mutation_rate': self.mutation_rate, 'crossover_rate': self.crossover_rate})

    def save_metrics(self, metrics):
        dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
        os.makedirs(dataset_folder, exist_ok=True)
        file_name = f"comparison_performance_ga.csv"
        file_path = os.path.join(dataset_folder, file_name)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(file_path, index=False)

