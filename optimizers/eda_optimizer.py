# eda_optimizer.py
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import plotly.graph_objects as go
from utils import save_convergence_history, save_performance_metrics
from optimization_utils import apply_adaptive_eda

class EDA_with_Batch:
    def __init__(self, pop_size, num_positions, num_plates, max_iter, mutation_rate, crossover_rate, lambda_1, lambda_2,
                 lambda_3, lambda_4, dataset_name, objectives, use_adaptive):
        self.pop_size = pop_size  # 种群大小
        self.num_positions = num_positions  # 库区/垛位数量
        self.num_plates = num_plates  # 钢板数量
        self.max_iter = max_iter  # 最大迭代次数
        self.lambda_1 = lambda_1  # 高度相关的权重参数
        self.lambda_2 = lambda_2  # 翻垛相关的权重参数
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.mutation_rate = mutation_rate  # 变异率
        self.crossover_rate = crossover_rate  # 交叉率
        self.dataset_name = dataset_name  # 数据集名称
        self.population = np.random.randint(0, num_positions, size=(pop_size, num_plates))  # 随机初始化种群
        self.best_position = None  # 最佳解
        self.best_score = np.inf  # 最优得分
        self.worst_score = -np.inf  # 最差得分
        self.convergence_data = []  # 用于保存收敛数据
        self.start_time = None  # 记录开始时间
        self.objectives = objectives  # OptimizationObjectives 实例
        self.use_adaptive = use_adaptive  # 是否使用自适应参数
        self.adaptive_param_data = []  # 用于保存自适应参数

        # Streamlit 占位符
        self.convergence_plot_placeholder = st.empty()
        self.adaptive_param_plot_placeholder = st.empty()

    def optimize(self):
        # 提供优化开始信息并显示运行时加载提示
        st.info("EDA Optimization started...")
        with st.spinner("Running EDA Optimization..."):
            self.start_time = time.time()  # 记录优化开始时间

            for iteration in range(self.max_iter):
                # 估计概率分布
                probabilities = self.estimate_distribution()

                # 使用概率分布生成新的种群
                new_population = self.generate_new_population(probabilities)

                # 选择操作：评估新种群并选择表现最好的个体
                for individual in new_population:
                    score = self.calculate_fitness(individual)

                    if score < self.best_score:
                        self.best_score = score
                        self.best_position = individual

                    if score > self.worst_score:
                        self.worst_score = score

                # 更新种群
                self.population = np.copy(new_population)

                # 如果使用自适应参数调节
                if self.use_adaptive:
                    self.mutation_rate, self.crossover_rate = apply_adaptive_eda(
                        self.mutation_rate, self.crossover_rate, self.best_score, self.use_adaptive)
                    self.record_adaptive_params()

                    # 每代更新自适应参数调节图
                    self.update_adaptive_param_plot()

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)

                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            # 计算优化结束时间
            time_elapsed = time.time() - self.start_time
            # 保存性能指标
            self.save_performance_metrics(time_elapsed)

            # 优化结束后，保存历史收敛数据
            history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "EDA")
            save_convergence_history(self.convergence_data, "EDA", self.dataset_name, history_data_dir)

    def estimate_distribution(self):
        # 估计种群的概率分布
        probabilities = np.zeros((self.num_plates, self.num_positions))

        # 统计每个位置被选择的频率
        for i in range(self.num_plates):
            for individual in self.population:
                probabilities[i, individual[i]] += 1

        # 将频率转换为概率
        probabilities = probabilities / self.pop_size
        return probabilities

    def generate_new_population(self, probabilities):
        # 根据估计的概率分布生成新种群
        new_population = []
        for _ in range(self.pop_size):
            new_individual = []
            for i in range(self.num_plates):
                new_position = np.random.choice(self.num_positions, p=probabilities[i])
                new_individual.append(new_position)
            new_population.append(np.array(new_individual))
        return np.array(new_population)

    def calculate_fitness(self, individual):
        # 计算个体的适应度得分
        combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(individual)
        energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
        balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
        space_utilization = self.objectives.maximize_space_utilization_v3(individual)

        score = (self.lambda_1 * combined_movement_turnover_penalty +
                 self.lambda_2 * energy_time_penalty +
                 self.lambda_3 * balance_penalty -
                 self.lambda_4 * space_utilization)

        return np.sum(score)  # 确保返回标量值

    def update_convergence_plot(self, current_iteration):
        # 动态更新收敛曲线
        iteration_data = [x[0] for x in self.convergence_data]
        score_data = [x[1] for x in self.convergence_data]

        # 使用 Plotly 绘制收敛曲线
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'))
        fig.update_layout(title=f'Convergence Curve - Iteration {current_iteration}',
                          xaxis_title='Iterations', yaxis_title='Best Score')

        self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

    def record_adaptive_params(self):
        self.adaptive_param_data.append({
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        })

    def update_adaptive_param_plot(self):
        # 绘制自适应参数变化图
        iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
        mutation_rate_data = [x['mutation_rate'] for x in self.adaptive_param_data]
        crossover_rate_data = [x['crossover_rate'] for x in self.adaptive_param_data]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=iteration_data, y=mutation_rate_data, mode='lines+markers', name='Mutation Rate'))
        fig.add_trace(
            go.Scatter(x=iteration_data, y=crossover_rate_data, mode='lines+markers', name='Crossover Rate'))

        fig.update_layout(title="Adaptive Parameter Changes", xaxis_title="Iterations",
                          yaxis_title="Parameter Values")

        self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

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
        file_name = f"comparison_performance_eda.csv"
        file_path = os.path.join(dataset_folder, file_name)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(file_path, index=False)

    def get_stable_iterations(self):
        stable_threshold = 1e-3  # 可以根据具体情况调整
        stable_iterations = 0
        for i in range(1, len(self.convergence_data)):
            if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                stable_iterations += 1
        return stable_iterations
