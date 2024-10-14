import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import save_convergence_history, save_performance_metrics
from optimization_utils import evaluate_parallel, evaluate_with_cache, apply_adaptive_pso, apply_adaptive_sa



# PSO 的粒子类定义
class Particle:
    def __init__(self, num_positions, num_plates):
        self.position = np.random.randint(0, num_positions, size=num_plates)  # 随机初始化位置
        self.velocity = np.zeros(num_plates)
        self.best_position = self.position.copy()
        self.best_score = np.inf

    def update_velocity(self, gbest_position, w, c1, c2, num_plates):
        r1 = np.random.rand(num_plates)
        r2 = np.random.rand(num_plates)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, num_positions):
        self.position = np.clip(self.position + self.velocity, 0, num_positions - 1).astype(int)



class PSO_with_Batch:
    def __init__(self, num_particles, num_positions, num_plates, w, c1, c2, max_iter, lambda_1, lambda_2, lambda_3,
                 lambda_4,
                 dataset_name, objectives, use_adaptive):
        self.num_particles = num_particles
        self.num_positions = num_positions
        self.num_plates = num_plates  # 添加 num_plates 参数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.use_adaptive = use_adaptive
        self.particles = [Particle(num_positions, num_plates) for _ in range(self.num_particles)]  # 传递 num_plates
        self.best_position = None
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.best_improvement = 0
        self.total_improvement = 0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.convergence_data = []  # 收敛数据
        self.dataset_name = dataset_name
        self.stable_iterations = 0
        self.prev_best_score = np.inf
        self.start_time = None
        self.objectives = objectives
        self.adaptive_param_data = []

    def optimize(self):
        self.start_time = time.time()

        # 创建进度条
        progress_bar = st.progress(0)
        with st.spinner("Running PSO Optimization..."):
            for iteration in range(self.max_iter):
                improvement_flag = False
                for particle in self.particles:
                    # 计算当前粒子的得分
                    current_score = self.evaluate_particle(particle)

                    if current_score < particle.best_score:
                        particle.best_score = current_score
                        particle.best_position = particle.position.copy()

                    if current_score < self.best_score:
                        improvement_flag = True
                        self.best_improvement = max(self.best_improvement, self.best_score - current_score)
                        self.best_score = current_score
                        self.best_position = particle.position.copy()

                    if current_score > self.worst_score:
                        self.worst_score = current_score

                if improvement_flag:
                    self.total_improvement += self.prev_best_score - self.best_score
                    self.prev_best_score = self.best_score
                    self.stable_iterations = 0
                else:
                    self.stable_iterations += 1

                # 更新粒子的位置和速度
                for particle in self.particles:
                    particle.update_velocity(self.best_position, self.w, self.c1, self.c2, self.num_plates)
                    particle.update_position(self.num_positions)

                # 自适应调节
                if self.use_adaptive:
                    self.w, self.c1, self.c2 = apply_adaptive_pso(self.w, self.c1, self.c2,
                                                                  self.best_score - current_score,
                                                                  self.use_adaptive)
                    self.record_adaptive_params()

                # 更新收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])

                # 更新进度条
                progress_percentage = (iteration + 1) / self.max_iter
                progress_bar.progress(progress_percentage)

        # 优化结束后清除进度条
        progress_bar.empty()

        time_elapsed = time.time() - self.start_time
        self.save_metrics(time_elapsed)

        # 优化结束后，保存历史收敛数据
        history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "PSO")
        save_convergence_history(self.convergence_data, "PSO", self.dataset_name, history_data_dir)

    def evaluate_particle(self, particle):
        combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
            particle.position)
        energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(particle.position)
        balance_penalty = self.objectives.maximize_inventory_balance_v2(particle.position)
        space_utilization = self.objectives.maximize_space_utilization_v3(particle.position)

        score = (self.lambda_1 * combined_movement_turnover_penalty +
                 self.lambda_2 * energy_time_penalty +
                 self.lambda_3 * balance_penalty -
                 self.lambda_4 * space_utilization)
        return score

    def record_adaptive_params(self):
        self.adaptive_param_data.append({'w': self.w, 'c1': self.c1, 'c2': self.c2})

    def save_metrics(self, time_elapsed):
        iterations = len(self.convergence_data)
        save_performance_metrics(
            self.best_score, self.worst_score, self.best_improvement, self.total_improvement,
            iterations, time_elapsed, self.convergence_data, self.stable_iterations, self.dataset_name, "PSO"
        )


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

        with st.spinner("Running SA Optimization..."):
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

class PSO_SA_Optimizer:
    def __init__(self, num_particles, num_positions, num_plates, w, c1, c2, max_iter_pso,
                 initial_temperature, cooling_rate, min_temperature, max_iterations_sa,
                 lambda_1, lambda_2, lambda_3, lambda_4, dataset_name, objectives, use_adaptive):
        # 保存 dataset_name 为类的属性
        self.dataset_name = dataset_name

        # 初始化 PSO 和 SA 参数
        self.pso_optimizer = PSO_with_Batch(
            num_particles=num_particles,
            num_positions=num_positions,
            num_plates=num_plates,  # 添加 num_plates 参数
            w=w, c1=c1, c2=c2, max_iter=max_iter_pso,
            lambda_1=lambda_1, lambda_2=lambda_2,
            lambda_3=lambda_3, lambda_4=lambda_4,
            dataset_name=dataset_name,
            objectives=objectives,
            use_adaptive=use_adaptive
        )

        self.sa_optimizer = SA_with_Batch(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            max_iterations=max_iterations_sa,
            lambda_1=lambda_1, lambda_2=lambda_2,
            lambda_3=lambda_3, lambda_4=lambda_4,
            num_positions=num_positions,
            num_plates=num_plates,  # 同样添加 num_plates 参数
            dataset_name=dataset_name,
            objectives=objectives,
            use_adaptive=use_adaptive
        )

        self.best_position = None  # 初始化 best_position 属性
        self.best_score = None  # 保存最终的 best_score
        self.convergence_data_pso_sa = []  # 存储混合优化的收敛数据
        self.start_time = None

    def optimize(self):
        self.start_time = time.time()

        # 运行 PSO 优化
        self.pso_optimizer.optimize()

        # 获取 PSO 最优解，作为 SA 初始解
        initial_position_for_sa = self.pso_optimizer.best_position

        # 运行 SA 进行局部优化
        best_position_sa, best_score_sa = self.sa_optimizer.optimize_from_position(initial_position_for_sa)

        # 保存最终的最佳位置和得分
        self.best_position = best_position_sa
        self.best_score = best_score_sa

        # 保存收敛数据
        self.convergence_data_pso_sa.extend(self.pso_optimizer.convergence_data)
        self.convergence_data_pso_sa.extend(self.sa_optimizer.convergence_data)

        # 保存收敛数据和性能指标
        self.save_convergence_data_pso_sa()
        time_elapsed = time.time() - self.start_time
        self.save_performance_metrics(time_elapsed, best_score_sa)

        return self.best_position, self.best_score

    def save_convergence_data_pso_sa(self):
        # 保存收敛数据
        dataset_folder = self.dataset_name.split('.')[0]  # 确保 dataset_name 已被赋值
        convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
        os.makedirs(convergence_data_dir, exist_ok=True)

        convergence_data_df = pd.DataFrame(self.convergence_data_pso_sa, columns=['Iteration', 'Best Score'])
        convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_psosa.csv')
        convergence_data_df.to_csv(convergence_data_path, index=False)

    def save_performance_metrics(self, time_elapsed, best_score_sa):
        # 保存性能指标
        iterations = len(self.convergence_data_pso_sa)
        worst_score = max([data[1] for data in self.convergence_data_pso_sa])
        best_improvement = abs(worst_score - best_score_sa)
        average_improvement = best_improvement / iterations if iterations > 0 else 0
        convergence_rate_value = (self.convergence_data_pso_sa[-1][1] - self.convergence_data_pso_sa[0][
            1]) / iterations
        relative_error_value = abs(best_score_sa) / (abs(best_score_sa) + 1e-6)
        stable_iterations = self.get_stable_iterations()

        metrics = {
            'Best Score': best_score_sa,
            'Worst Score': worst_score,
            'Best Improvement': best_improvement,
            'Average Improvement': average_improvement,
            'Iterations': iterations,
            'Time (s)': time_elapsed,
            'Convergence Rate': convergence_rate_value,
            'Relative Error': relative_error_value,
            'Convergence Speed (Stable Iterations)': stable_iterations
        }

        dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
        os.makedirs(dataset_folder, exist_ok=True)
        file_path = os.path.join(dataset_folder, 'comparison_performance_psosa.csv')

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(file_path, index=False)

    def get_stable_iterations(self):
        # 获取稳定的迭代次数
        stable_threshold = 1e-3
        stable_iterations = 0
        for i in range(1, len(self.convergence_data_pso_sa)):
            if abs(self.convergence_data_pso_sa[i][1] - self.convergence_data_pso_sa[i - 1][1]) < stable_threshold:
                stable_iterations += 1
        return stable_iterations
