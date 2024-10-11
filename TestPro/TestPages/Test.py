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

# 堆垛平衡度计算函数（标准差越低，均衡度越好）
def calculate_balance(stack_heights):
    avg_heights = np.mean(stack_heights, axis=1)  # 每个库区的平均高度
    balance_score = np.std(avg_heights)  # 计算标准差衡量库区堆垛均衡度
    return balance_score


# 实现人工堆垛规则
def manual_stacking(df, area_positions, stack_dimensions):
    num_areas = len(area_positions)  # 库区数量
    num_stacks_per_area = len(stack_dimensions[0])  # 每个库区的垛位数量

    # 初始化堆垛高度
    stack_heights = np.zeros((num_areas, num_stacks_per_area))

    # 按批次分组钢板
    grouped = df.groupby('Batch')

    for name, group in grouped:
        st.write(f"处理批次: {name}")
        for _, plate in group.iterrows():
            # 查找高度最低的垛位
            area, stack = find_lowest_stack(stack_heights)

            # 更新当前垛位的高度，增加钢板厚度
            stack_heights[area, stack] += plate['Thickness']

    # 计算均衡度
    balance_score = calculate_balance(stack_heights)

    return stack_heights, balance_score


# 查找当前高度最低的垛位
def find_lowest_stack(stack_heights):
    min_height = np.min(stack_heights)
    area, stack = np.where(stack_heights == min_height)

    # 返回找到的最低垛位（如果有多个，随机选择一个）
    selected_area = area[0]
    selected_stack = stack[0]

    return selected_area, selected_stack


# Streamlit 页面配置
st.set_page_config(page_title="人工堆垛规则", page_icon="⚙")

st.title("Steel Plate Manual Stacking")

# 获取数据集
data_choice = st.selectbox("Choose dataset", ("Use system dataset", "Upload your own dataset"))
df = None

if data_choice == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload your steel plate dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset:")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset to proceed.")
else:
    # 系统数据集加载示例
    system_data_dir = "data"
    system_dataset_path = os.path.join(system_data_dir, "system_dataset.csv")
    df = pd.read_csv(system_dataset_path)
    st.write(f"Using system dataset:")
    st.write(df.head())

# 仓库和堆叠配置
st.write("Warehouse and Stack Configuration")
use_default_config = st.checkbox("Use default warehouse and stack configuration", value=True)

if not use_default_config:
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
            width = st.number_input(f"Stack {stack + 1} width (mm)", 1000, 20000, 6000,
                                    key=f'stack_width_area_{area}_{stack}')
            length = st.number_input(f"Stack {stack + 1} length (mm)", 1000, 20000, 3000,
                                     key=f'stack_length_area_{area}_{stack}')
            area_stack_positions.append((x, y))
            area_stack_dimensions.append((length, width))
        area_positions[area] = area_stack_positions
        stack_dimensions[area] = area_stack_dimensions
else:

    area_positions = DEFAULT_AREA_POSITIONS
    stack_dimensions = DEFAULT_STACK_DIMENSIONS

# 执行人工堆垛
if df is not None:
    st.write("### Manual Stacking Results")
    stack_heights, balance_score = manual_stacking(df, area_positions, stack_dimensions)

    st.write("堆垛高度：")
    st.write(stack_heights)

    st.write(f"库区堆垛均衡度（标准差，越低越好）: {balance_score:.2f}")

    # 画堆垛高度分布图
    st.write("### 堆垛高度分布图")
    fig, ax = plt.subplots()
    im = ax.imshow(stack_heights, cmap="Blues")

    # 在每个方格内显示高度数值
    for i in range(stack_heights.shape[0]):
        for j in range(stack_heights.shape[1]):
            ax.text(j, i, f'{stack_heights[i, j]:.1f}', ha='center', va='center', color='black')

    plt.title('Stack Heights by Area and Position')
    st.pyplot(fig)

