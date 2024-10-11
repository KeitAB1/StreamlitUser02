import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 从 constants 文件中引入常量
from constants import OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR

# Streamlit 页面配置
st.set_page_config(page_title="Optimization Application", page_icon="⚙")

# 创建用于保存图像的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 将选择应用场景移至左侧栏
with st.sidebar:
    st.write("Select Application Scenario")
    application_scenario = st.selectbox("Choose the application scenario", ("Steel Plate Stacking", "Container Loading"))

# 根据选择的应用场景加载对应的参数配置
if application_scenario == "Steel Plate Stacking":
    st.title("Steel Plate Stacking Optimization")
    from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives

elif application_scenario == "Container Loading":
    st.title("Container Loading Optimization")
    from optimization_objectives import ContainerLoadingObjectives

    # 数据集选择
    data_choice = st.selectbox("Choose dataset", ("Use system dataset", "Upload your own dataset"))
    df = None
    dataset_name = None

    if data_choice == "Upload your own dataset":
        uploaded_file = st.file_uploader("Upload your container dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            dataset_name = uploaded_file.name.split('.')[0]
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded dataset:")
            st.write(df.head())
        else:
            st.warning("Please upload a dataset to proceed.")
    elif data_choice == "Use system dataset":
        system_data_dir = "data/container_loading_dataset"
        available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]
        selected_dataset = st.selectbox("Select a system dataset", [""] + available_datasets)
        if selected_dataset:
            dataset_name = selected_dataset
            system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
            df = pd.read_csv(system_dataset_path)
            st.write(f"Using system dataset: {selected_dataset}")
            st.write(df.head())
        else:
            st.warning("Please select a system dataset to proceed.")

    # 选择算法
    with st.sidebar:
        algorithms = ["SA (Simulated Annealing)"]
        selected_algorithm = st.selectbox("Select Optimization Algorithm", algorithms)
        use_adaptive = st.checkbox("Use Adaptive Parameter Adjustment", value=False)

        if selected_algorithm == "SA (Simulated Annealing)":
            st.subheader("SA Parameters")
            initial_temperature = st.number_input("Initial Temperature", value=1000.0)
            cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
            min_temperature = st.number_input("Minimum Temperature", value=0.1)
            max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)

    # 优化分析
    if df is not None:
        # 提取集装箱和物品信息
        items = df[['Length (cm)', 'Width (cm)', 'Height (cm)', 'Weight (kg)']].values
        container_dimensions = (600, 250, 240)  # 设定集装箱的尺寸（长，宽，高）

        objectives = ContainerLoadingObjectives(items, container_dimensions)

        class SA_with_ContainerLoading:
            def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, objectives):
                self.initial_temperature = initial_temperature
                self.cooling_rate = cooling_rate
                self.min_temperature = min_temperature
                self.max_iterations = max_iterations
                self.objectives = objectives

                self.best_position = None
                self.best_score = np.inf
                self.convergence_data = []
                self.temperature_data = []
                self.start_time = None

                # 用于实时绘制收敛曲线的占位符
                self.convergence_plot_placeholder = st.empty()

            def evaluate(self, position):
                return self.objectives.evaluate(position)

            def optimize(self):
                initial_position = np.random.rand(self.objectives.num_items, 3) * np.array(
                    self.objectives.container_dimensions)
                return self.optimize_from_position(initial_position)

            def optimize_from_position(self, initial_position):
                current_temperature = self.initial_temperature
                current_position = initial_position
                current_score = self.evaluate(current_position)

                self.best_position = current_position.copy()
                self.best_score = current_score
                self.start_time = time.time()

                st.info("SA Optimization for Container Loading started...")
                with st.spinner("Running SA Optimization for Container Loading..."):
                    for iteration in range(self.max_iterations):
                        if current_temperature < self.min_temperature:
                            break

                        # 生成新解
                        new_position = current_position + (np.random.rand(*current_position.shape) - 0.5) * 10
                        new_position = np.clip(new_position, 0, self.objectives.container_dimensions)

                        # 评估新解
                        new_score = self.evaluate(new_position)

                        delta = new_score - current_score
                        if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                            current_position = new_position
                            current_score = new_score

                        if current_score < self.best_score:
                            self.best_score = current_score
                            self.best_position = current_position.copy()

                        # 更新温度
                        current_temperature *= self.cooling_rate

                        self.convergence_data.append([iteration + 1, self.best_score])
                        self.temperature_data.append(current_temperature)

                        # 实时更新收敛曲线
                        self.update_convergence_plot()

                    st.success("SA Optimization for Container Loading complete!")

                return self.best_position, self.best_score

            def update_convergence_plot(self):
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
                    title='Convergence Curve',
                    xaxis_title='Iterations',
                    legend=dict(x=0.75, y=1)
                )
                fig.update_yaxes(title_text="Best Score", secondary_y=False)
                fig.update_yaxes(title_text="Temperature", secondary_y=True)

                self.convergence_plot_placeholder.plotly_chart(fig)

            def visualize_container_loading_2d(self):
                fig = go.Figure()

                # 设定集装箱的尺寸
                container_length, container_width, _ = self.objectives.container_dimensions

                # 绘制每个物品的矩形
                for i, item in enumerate(self.objectives.items):
                    length, width, _, _ = item[:4]
                    x_offset, y_offset, _ = self.best_position[i]

                    fig.add_trace(go.Scatter(
                        x=[x_offset, x_offset + length, x_offset + length, x_offset, x_offset],
                        y=[y_offset, y_offset, y_offset + width, y_offset + width, y_offset],
                        fill='toself',
                        mode='lines',
                        name=f'Item {i + 1}',
                        opacity=0.6
                    ))

                fig.update_layout(
                    xaxis_title='Length (cm)',
                    yaxis_title='Width (cm)',
                    title='2D Visualization of Container Loading',
                    width=800,
                    height=600
                )
                st.plotly_chart(fig)

            def compute_and_display_results(self):
                # 计算和显示利用率和重心偏移的表格
                container_volume = np.prod(self.objectives.container_dimensions)
                total_volume = sum([item[0] * item[1] * item[2] for item in self.objectives.items])
                volume_utilization = (total_volume / container_volume) * 100

                max_weight = self.objectives.max_weight  # 使用正确的属性名称
                total_weight = sum([item[3] for item in self.objectives.items])
                weight_utilization = (total_weight / max_weight) * 100

                # 计算重心偏移量
                container_length, container_width, container_height = self.objectives.container_dimensions
                total_weight = sum([item[3] for item in self.objectives.items])

                weighted_x_sum = sum([(self.best_position[i][0] + item[0] / 2) * item[3] for i, item in
                                      enumerate(self.objectives.items)])
                weighted_y_sum = sum([(self.best_position[i][1] + item[1] / 2) * item[3] for i, item in
                                      enumerate(self.objectives.items)])
                weighted_z_sum = sum([(self.best_position[i][2] + item[2] / 2) * item[3] for i, item in
                                      enumerate(self.objectives.items)])

                if total_weight == 0:
                    st.warning("Total weight is zero, unable to compute center of gravity offset.")
                    return

                center_of_gravity = (
                    weighted_x_sum / total_weight,
                    weighted_y_sum / total_weight,
                    weighted_z_sum / total_weight
                )
                ideal_center_of_gravity = (
                    container_length / 2,
                    container_width / 2,
                    container_height / 2
                )
                center_of_gravity_offset = np.linalg.norm(
                    np.array(center_of_gravity) - np.array(ideal_center_of_gravity)
                )

                # 显示结果表格
                results_df = pd.DataFrame({
                    "Metric": ["Volume Utilization (%)", "Weight Utilization (%)", "Center of Gravity Offset (cm)"],
                    "Value": [round(volume_utilization, 2), round(weight_utilization, 2),
                              round(center_of_gravity_offset, 2)]
                })

                st.write("### Optimization Results")
                st.table(results_df)


        # 如果选择了模拟退火算法进行优化
        if selected_algorithm == "SA (Simulated Annealing)":
            optimizer = SA_with_ContainerLoading(
                initial_temperature,
                cooling_rate,
                min_temperature,
                max_iterations_sa,
                objectives
            )
            best_position, best_score = optimizer.optimize()

            st.write("Best Position (Container Loading):")
            st.write(best_position)
            st.write(f"Best Score: {best_score}")

            # 调用可视化方法，展示2D装载结果
            optimizer.visualize_container_loading_2d()

            # 计算并展示优化结果的表格
            optimizer.compute_and_display_results()
