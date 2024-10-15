import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
# 从 PSOSA 优化器引入 PSO_SA_Optimizer
from optimizers.psosa_optimizer import PSO_SA_Optimizer
from optimizers.eda_optimizer import EDA_with_Batch  # 引入 EDA 优化算法
from optimizers.ga_optimizer import GA_with_Batch  # 引入 GA 优化算法
from optimizers.co_ea_optimizer import CoEA_with_Batch  # 引入 CoEA 优化算法
from utils import save_convergence_history, add_download_button, run_optimization, display_icon_with_header
from optimizer_runner import OptimizerRunner  # 导入优化算法管理器

# 从 constants 文件中引入常量
from constants import (
    OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH,
    AREA_POSITIONS_DIMENSIONS,
    DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS,
    HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE,
    INBOUND_POINT, OUTBOUND_POINT, Dki
)

# Streamlit 页面配置
st.set_page_config(page_title="智能钢板堆垛系统", page_icon="⚙", layout="wide")

# 创建用于保存图像的目录
output_dir_base = "result/"
os.makedirs(output_dir_base, exist_ok=True)

st.title("⚙ 智能钢板堆垛系统")

# 使用 display_icon_with_header 函数替换现有的图标和标题显示逻辑
display_icon_with_header("data/icon/icon01.jpg", "数据导入", font_size="24px", icon_size="20px")

# 使用 display_icon_with_header 函数替换部分的展示
col3, col4, col11 = st.columns([0.01, 0.25, 0.55])
with col3:
    st.image("data/icon/icon02.jpg", width=20)
with col4:
    data_choice = st.selectbox("选择数据集", ["使用系统数据集", "上传自定义数据集"])


df = None
dataset_name = None
system_data_dir = "data/Steel_Data"

# 导入数据集的逻辑
if data_choice == "上传自定义数据集":
    uploaded_file = st.file_uploader("上传钢板数据集 (CSV)", type=["csv"])
    if uploaded_file:
        dataset_name = uploaded_file.name.split('.')[0]
        df = pd.read_csv(uploaded_file)
        st.write("已上传的数据集：")
        st.write(df.head())
    else:
        st.warning("请上传数据集以继续。")
elif data_choice == "使用系统数据集":
    # 创建两列布局，分别放置选择数据集和选择优化模式
    col7, col5, col8, col6, col9 = st.columns([0.01, 0.2, 0.01, 0.1, 0.3])

    with col7:
        st.image("data/icon/icon02.jpg", width=20)
    # 左侧列：选择系统数据集
    with col5:
        available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]
        selected_dataset = st.selectbox("选择系统数据集", [""] + available_datasets)
        if selected_dataset:
            dataset_name = selected_dataset
            system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
            df = pd.read_csv(system_dataset_path)
    with col8:
        st.image("data/icon/icon02.jpg", width=20)
    # 右侧列：选择优化模式
    with col6:
        optimization_mode = st.selectbox("选择优化模式", ["普通优化", "深度优化"])


start_work = st.button("Start Work")


# 优化参数配置
initial_temperature = 1000.0
cooling_rate = 0.9
min_temperature = 0.1
max_iterations_sa = 1
num_particles = 30  # 粒子群大小
max_iter_pso = 1  # PSO最大迭代次数
w, c1, c2 = 0.5, 1.5, 1.5  # PSO 参数
lambda_1, lambda_2, lambda_3, lambda_4 = 1.0, 1.0, 1.0, 1.0
use_adaptive = True

# EDA 优化参数配置
pop_size = 50  # EDA 种群大小
max_iter_eda = 1  # EDA最大迭代次数
mutation_rate = 0.1  # EDA变异率
crossover_rate = 0.7  # EDA交叉率

# GA 优化参数配置
ga_population_size = 50
ga_generations = 1  # GA最大迭代次数
ga_mutation_rate = 0.1
ga_crossover_rate = 0.8

# CoEA 优化参数配置
coea_population_size = 50  # CoEA 种群大小
coea_generations = 1  # CoEA 最大迭代次数
coea_mutation_rate = 0.1  # CoEA 变异率
coea_crossover_rate = 0.8  # CoEA 交叉率

# 优化分析
if df is not None:
    output_dir = os.path.join(output_dir_base, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # 数据准备
    plates = df[['Length', 'Width', 'Thickness', 'Material_Code', 'Batch', 'Entry Time', 'Delivery Time']].values
    num_positions = len(Dki)
    num_plates = len(plates)
    heights = np.zeros(num_positions)
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values
    batches = df['Batch'].values

    objectives = OptimizationObjectives(
        plates=plates,
        heights=heights,
        delivery_times=delivery_times,
        batches=batches,
        Dki=Dki,
        area_positions=DEFAULT_AREA_POSITIONS,
        inbound_point=INBOUND_POINT,
        outbound_point=OUTBOUND_POINT,
        horizontal_speed=HORIZONTAL_SPEED,
        vertical_speed=VERTICAL_SPEED
    )

    # 多种优化算法的参数配置
    algorithms_params = {
        "PSO_SA_Optimizer": {
            'num_particles': num_particles,
            'num_positions': num_positions,
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter_pso': max_iter_pso,
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations_sa': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': num_positions,  # 库区/垛位数量
            'num_plates': num_plates,  # 钢板数量
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        },
        "EDA_with_Batch": {
            'pop_size': pop_size,
            'num_positions': num_positions,
            'num_plates': num_plates,  # 添加 num_plates 参数
            'max_iter': max_iter_eda,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        },
        "GA_with_Batch": {
            'population_size': ga_population_size,
            'mutation_rate': ga_mutation_rate,
            'crossover_rate': ga_crossover_rate,
            'generations': ga_generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': num_positions,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'plates': plates,
            'delivery_times': delivery_times,
            'batches': batches,
            'use_adaptive': use_adaptive
        },
        "CoEA_with_Batch": {
            'population_size': coea_population_size,
            'mutation_rate': coea_mutation_rate,
            'crossover_rate': coea_crossover_rate,
            'generations': coea_generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': num_positions,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        }
    }

    if start_work:
        if optimization_mode == "深度优化":  # 判断用户是否选择深度优化模式
            # 启用深度优化，运行多个优化算法并选择最佳方案
            optimizer_runner = OptimizerRunner(algorithms_params, df, DEFAULT_AREA_POSITIONS, output_dir_base)
            optimizer_runner.run_optimization()
        else:
            # 仅运行单一优化算法 PSO_SA_Optimizer
            run_optimization(PSO_SA_Optimizer, algorithms_params["PSO_SA_Optimizer"], df, DEFAULT_AREA_POSITIONS, output_dir_base, "psosa")
