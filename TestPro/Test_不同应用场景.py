import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import time
from optimization_objectives import OptimizationObjectives
from utils import run_optimization, save_convergence_plot, save_performance_metrics
from optimizers.sa_optimizer import SA_with_Batch
from optimization_utils import evaluate_parallel, evaluate_with_cache, run_distributed_optimization
from optimization_utils import apply_adaptive_pso, apply_adaptive_sa, apply_adaptive_ga, apply_adaptive_coea, apply_adaptive_eda
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import logging  # 日志模块
from utils import save_convergence_history


# 从 constants 文件中引入常量
from constants import OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH
from constants import DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS, HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE, INBOUND_POINT, OUTBOUND_POINT, Dki

# 日志配置
# logging.basicConfig(filename="optimization.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Streamlit 页面配置
st.set_page_config(page_title="Steel Plate Stacking Optimization", page_icon="⚙")


# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.title("Steel Plate Stacking Optimization")

# 并行计算适应度函数
def evaluate_parallel(positions, evaluate_func):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_func, positions))
    return results

# 获取数据集
system_data_dir = "data"
available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]

st.write("Warehouse and Stack Configuration")
use_default_config = st.checkbox("Use default warehouse and stack configuration", value=True)

if not use_default_config:
    # 如果用户选择自定义配置，显示相关输入框
    num_areas = st.number_input("Number of Areas", 1, 10, 6)

    area_positions = {}
    stack_dimensions = {}
    for area in range(num_areas):
        st.write(f"### Area {area + 1}")
        num_stacks = st.number_input(f"Number of Stacks in Area {area + 1}", 1, 10, 4, key=f'num_stacks_area_{area}')
        area_stack_positions = []
        area_stack_dimensions = []
        for stack in range(num_stacks):
            x = st.number_input(f"Stack {stack + 1} X position", key=f'stack_x_area_{area}_{stack}')
            y = st.number_input(f"Stack {stack + 1} Y position", key=f'stack_y_area_{area}_{stack}')
            width = st.number_input(f"Stack {stack + 1} width (mm)", 1000, 20000, 6000, key=f'stack_width_area_{area}_{stack}')
            length = st.number_input(f"Stack {stack + 1} length (mm)", 1000, 20000, 3000, key=f'stack_length_area_{area}_{stack}')
            area_stack_positions.append((x, y))
            area_stack_dimensions.append((length, width))

        area_positions[area] = area_stack_positions
        stack_dimensions[area] = area_stack_dimensions
else:
    area_positions = DEFAULT_AREA_POSITIONS
    stack_dimensions = DEFAULT_STACK_DIMENSIONS

# 查看当前配置
if "show_stack_config" not in st.session_state:
    st.session_state["show_stack_config"] = False

if st.button("View/Hide Current Stack Configuration"):
    st.session_state["show_stack_config"] = not st.session_state["show_stack_config"]

if st.session_state["show_stack_config"]:
    # 堆叠区域位置信息表格
    st.write("### Current Area Positions")
    area_positions_data = []
    for area, positions in DEFAULT_AREA_POSITIONS.items():
        area_positions_data.append({
            "Area": f"Area {area + 1}",
            "Positions": str(positions)
        })

    positions_df = pd.DataFrame(area_positions_data)
    st.table(positions_df)


    # 堆叠尺寸信息表格
    st.write("### Current Stack Dimensions")
    stack_dimensions_data = []
    for area, dimensions in DEFAULT_STACK_DIMENSIONS.items():
        for idx, (length, width) in enumerate(dimensions):
            stack_dimensions_data.append({
                "Area": f"Area {area + 1}",
                "Stack": f"Stack {idx + 1}",
                "Length (mm)": length,
                "Width (mm)": width
            })

    dimensions_df = pd.DataFrame(stack_dimensions_data)
    st.table(dimensions_df)

# 数据集选择
data_choice = st.selectbox("Choose dataset", ("Use system dataset", "Upload your own dataset"))
df = None
dataset_name = None

if data_choice == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload your steel plate dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name.split('.')[0]
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset:")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset to proceed.")
elif data_choice == "Use system dataset":
    selected_dataset = st.selectbox("Select a system dataset", [""] + available_datasets)
    if selected_dataset:
        dataset_name = selected_dataset
        system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
        df = pd.read_csv(system_dataset_path)
        st.write(f"Using system dataset: {selected_dataset}")
        st.write(df.head())
    else:
        st.warning("Please select a system dataset to proceed.")


def get_optimization_weights():
    st.write("#### Optimize Target Weight")

    # 优化目标权重
    lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
    lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
    lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
    lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    return lambda_1, lambda_2, lambda_3, lambda_4


# 选择算法
with st.sidebar:
    algorithms = ["SA (Simulated Annealing)", "GA (Genetic Algorithm)", "PSO (Particle Swarm Optimization)",
                  "PSO + SA (Hybrid Optimization)", "ACO (Ant Colony Optimization)", "DE (Differential Evolution)",
                  "CoEA (Co-Evolutionary Algorithm)", "EDA (Estimation of Distribution Algorithm)",
                  "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)",
                  "HACO (Hybrid Ant Colony Optimization)", " ADE(Adaptive Differential Evolution)"]
    selected_algorithm = st.selectbox("Select Optimization Algorithm", algorithms)
    use_adaptive = st.checkbox("Use Adaptive Parameter Adjustment", value=False)

    if selected_algorithm == "SA (Simulated Annealing)":
        st.subheader("SA Parameters")
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)
        # 调用优化权重函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()


# 优化分析
if df is not None:
    output_dir_base = f"result/final_stack_distribution/{dataset_name}"
    os.makedirs(os.path.join(output_dir_base, 'final_stack_distribution_height'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_base, 'final_stack_distribution_plates'), exist_ok=True)

    plates = df[['Length', 'Width', 'Thickness', 'Material_Code', 'Batch', 'Entry Time', 'Delivery Time']].values
    plate_areas = plates[:, 0] * plates[:, 1]
    num_plates = len(plates)
    batches = df['Batch'].values

    heights = np.zeros(len(Dki))

    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values

    objectives = OptimizationObjectives(
        plates=plates,
        heights=heights,
        delivery_times=delivery_times,
        batches=batches,
        Dki=Dki,
        area_positions=area_positions,
        inbound_point=INBOUND_POINT,
        outbound_point=OUTBOUND_POINT,
        horizontal_speed=HORIZONTAL_SPEED,
        vertical_speed=VERTICAL_SPEED
    )

    class SA_with_Batch:
        def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, lambda_1, lambda_2,
                     lambda_3, lambda_4, num_positions, dataset_name, objectives, use_adaptive):
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
            self.worst_score = -np.inf  # 初始化最差得分
            self.convergence_data = []
            self.temperature_data = []
            self.adaptive_param_data = []
            self.start_time = None

            self.cache = {}
            self.score_changes = []  # 初始化得分变化列表

            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

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
                # logging.error(f"Error in evaluation: {e}")
                return np.inf

        def optimize(self):
            initial_position = np.random.randint(0, self.num_positions, size=num_plates)
            return self.optimize_from_position(initial_position)

        def optimize_from_position(self, initial_position):
            current_temperature = self.initial_temperature
            current_position = initial_position
            current_score = self.evaluate_with_cache(current_position)

            self.best_position = current_position.copy()
            self.best_score = current_score
            self.worst_score = current_score  # 设置初始最差得分
            self.start_time = time.time()

            # 初始化性能指标
            scores = []  # 所有得分
            unsuccessful_attempts = 0  # 失败次数

            st.info("SA Optimization started...")
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
                        unsuccessful_attempts += 1  # 记录未成功的尝试

                    if current_score < self.best_score:
                        self.best_score = current_score
                        self.best_position = current_position.copy()

                    if current_score > self.worst_score:
                        self.worst_score = current_score  # 更新最差得分

                    # 更新得分列表和得分变化
                    scores.append(current_score)
                    self.score_changes.append(delta)  # 使用实例变量

                    current_temperature, self.cooling_rate = apply_adaptive_sa(
                        current_temperature, self.cooling_rate, delta, self.use_adaptive)

                    if self.use_adaptive:
                        self.record_adaptive_params()

                    self.convergence_data.append([iteration + 1, self.best_score])
                    self.temperature_data.append(current_temperature)

                    self.update_convergence_plot(iteration + 1)
                    # logging.info(
                    #     f"Iteration {iteration + 1}/{self.max_iterations}, Best Score: {self.best_score}, Temperature: {current_temperature}")

                st.success("Optimization complete!")

            # 计算平均得分和标准差
            avg_score = np.mean(scores)
            score_std = np.std(scores)
            total_attempts = len(scores)

            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed, avg_score, score_std, unsuccessful_attempts)  # 保存性能指标

            # 优化结束后，保存历史收敛数据
            history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "SA")
            save_convergence_history(self.convergence_data, "SA", self.dataset_name, history_data_dir)

            return self.best_position, self.best_score

        def record_adaptive_params(self):
            self.adaptive_param_data.append({'cooling_rate': self.cooling_rate})

        def update_convergence_plot(self, current_iteration):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]
            temperature_data = self.temperature_data

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=iteration_data, y=temperature_data, mode='lines+markers', name='Temperature',
                           line=dict(dash='dash')),
                secondary_y=True
            )

            fig.update_layout(
                title=f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}',
                xaxis_title='Iterations',
                legend=dict(x=0.75, y=1)
            )

            fig.update_yaxes(title_text="Best Score", secondary_y=False)
            fig.update_yaxes(title_text="Temperature", secondary_y=True)

            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

            if self.use_adaptive:
                self.update_adaptive_param_plot()

        def update_adaptive_param_plot(self):
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            cooling_rate_data = [x['cooling_rate'] for x in self.adaptive_param_data]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iteration_data, y=cooling_rate_data, mode='lines+markers', name='Cooling Rate')
            )
            fig.update_layout(
                title="Adaptive Parameter Changes",
                xaxis_title="Iterations",
                yaxis_title="Cooling Rate",
                legend=dict(x=0.75, y=1)
            )

            self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

        def save_metrics(self, time_elapsed, avg_score, score_std, unsuccessful_attempts):
            iterations = len(self.convergence_data)
            total_improvement = np.sum(self.score_changes)  # 使用实例变量
            self.worst_score = max(self.score_changes)  # 更新最差得分

            save_performance_metrics(
                self.best_score,
                self.worst_score,
                total_improvement,
                total_improvement,  # 或者替换为您需要的其他参数
                iterations,
                time_elapsed,
                self.convergence_data,
                len(self.adaptive_param_data),
                self.dataset_name,
                "SA"
            )

    if selected_algorithm == "SA (Simulated Annealing)":
        sa_params = {
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        }

        run_optimization(SA_with_Batch, sa_params, df, area_positions, output_dir_base, "sa")

