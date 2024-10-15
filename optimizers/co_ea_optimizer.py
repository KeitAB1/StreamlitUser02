import numpy as np
import pandas as pd
import os
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from utils import save_convergence_history
from optimization_utils import apply_adaptive_pso, apply_adaptive_sa, apply_adaptive_ga, apply_adaptive_coea, apply_adaptive_eda

class CoEA_with_Batch:
    def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                 lambda_4, num_positions, dataset_name, objectives, use_adaptive=False):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.num_positions = num_positions
        self.dataset_name = dataset_name
        self.objectives = objectives  # 优化目标实例
        self.use_adaptive = use_adaptive
        self.adaptive_param_data = []

        # 创建多个子种群，假设分为两个子种群
        self.subpopulations = [
            [np.random.randint(0, num_positions, size=len(objectives.plates)) for _ in range(population_size // 2)] for _ in
            range(2)
        ]

        self.best_position = None
        self.best_score = np.inf
        self.worst_score = -np.inf  # 最差得分
        self.convergence_data = []
        self.start_time = None  # 记录运行时间

    def fitness(self, individual):
        # 计算适应度得分
        combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
            individual)
        energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
        balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
        space_utilization = self.objectives.maximize_space_utilization_v3(individual)

        score = (self.lambda_1 * combined_movement_turnover_penalty +
                 self.lambda_2 * energy_time_penalty +
                 self.lambda_3 * balance_penalty -
                 self.lambda_4 * space_utilization)

        return np.sum(score)  # 返回标量值

    def select(self, subpopulation):
        fitness_scores = np.array([self.fitness(individual) for individual in subpopulation])
        probabilities = np.exp(-fitness_scores / np.sum(fitness_scores))
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(len(subpopulation), size=len(subpopulation), p=probabilities)
        return [subpopulation[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
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
        # st.info("CoEA Optimization started...")  # 提供优化开始信息
        with st.spinner("Running CoEA Optimization..."):  # 提供运行时加载提示
            self.start_time = time.time()  # 记录优化开始时间

            progress_bar = st.progress(0)  # 初始化进度条

            for generation in range(self.generations):
                for subpopulation in self.subpopulations:
                    new_subpopulation = []
                    selected_subpopulation = self.select(subpopulation)

                    # 交叉与变异操作使用多线程并行处理
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        for i in range(0, len(subpopulation), 2):
                            parent1 = selected_subpopulation[i]
                            parent2 = selected_subpopulation[min(i + 1, len(subpopulation) - 1)]
                            futures.append(executor.submit(self.crossover, parent1, parent2))

                        for future in futures:
                            child1, child2 = future.result()
                            new_subpopulation.append(self.mutate(child1))
                            new_subpopulation.append(self.mutate(child2))

                    subpopulation[:] = new_subpopulation

                # 协同进化，交换子种群个体
                self.exchange_subpopulations()

                # 更新全局最优解
                best_individual_gen = min([min(subpop, key=self.fitness) for subpop in self.subpopulations],
                                          key=self.fitness)
                best_score_gen = self.fitness(best_individual_gen)

                if best_score_gen < self.best_score:
                    self.best_score = best_score_gen
                    self.best_position = best_individual_gen.copy()

                if best_score_gen > self.worst_score:
                    self.worst_score = best_score_gen

                # 记录自适应参数
                if self.use_adaptive:
                    self.mutation_rate, self.crossover_rate = apply_adaptive_coea(
                        self.mutation_rate, self.crossover_rate, best_score_gen, self.best_score, self.use_adaptive)
                    self.record_adaptive_params()

                # 记录收敛数据
                self.convergence_data.append([generation + 1, self.best_score])

                # 更新进度条
                progress_percentage = (generation + 1) / self.generations
                progress_bar.progress(progress_percentage)

                # print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

            # 优化结束后清除进度条
            progress_bar.empty()

            time_elapsed = time.time() - self.start_time
            self.save_performance_metrics(time_elapsed)

            # 保存收敛历史数据
            history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "CoEA")
            save_convergence_history(self.convergence_data, "CoEA", self.dataset_name, history_data_dir)

    def exchange_subpopulations(self):
        exchange_size = self.population_size // 10  # 交换10%的个体
        for i in range(exchange_size):
            idx1 = np.random.randint(len(self.subpopulations[0]))
            idx2 = np.random.randint(len(self.subpopulations[1]))
            # 交换个体
            self.subpopulations[0][idx1], self.subpopulations[1][idx2] = self.subpopulations[1][idx2], \
                                                                         self.subpopulations[0][idx1]

    def record_adaptive_params(self):
        self.adaptive_param_data.append(
            {'mutation_rate': self.mutation_rate, 'crossover_rate': self.crossover_rate})

    def save_performance_metrics(self, time_elapsed):
        iterations = len(self.convergence_data)
        best_improvement = np.inf if iterations == 0 else abs(self.worst_score - self.best_score)
        average_improvement = np.inf if iterations == 0 else best_improvement / iterations
        convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
        relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)
        stable_iterations = self.get_stable_iterations()

        metrics = {
            'Best Score': self.best_score,
            'Worst Score': self.worst_score,
            'Best Improvement': best_improvement,
            'Average Improvement': average_improvement,
            'Iterations': iterations,
            'Time (s)': time_elapsed,
            'Convergence Rate': convergence_rate_value,
            'Relative Error': relative_error_value,
            'Convergence Speed (Stable Iterations)': stable_iterations
        }

        # 保存性能指标
        dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
        os.makedirs(dataset_folder, exist_ok=True)
        file_name = f"comparison_performance_coea.csv"
        file_path = os.path.join(dataset_folder, file_name)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(file_path, index=False)

    def get_stable_iterations(self):
        stable_threshold = 1e-3
        stable_iterations = 0
        for i in range(1, len(self.convergence_data)):
            if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                stable_iterations += 1
        return stable_iterations
