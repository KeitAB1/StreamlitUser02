import streamlit as st
import pandas as pd
import numpy as np
import os
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
from optimizers.psosa_optimizer import PSO_SA_Optimizer  # 添加 PSOSA 组合优化算法
from utils import run_optimization, display_icon_with_header
from optimizer_params import SA_PARAMS, PSO_PARAMS  # 引入 PSO 和 SA 参数

# 从 constants 文件中引入常量
from constants import (
    DEFAULT_AREA_POSITIONS, Dki, INBOUND_POINT, OUTBOUND_POINT,
    HORIZONTAL_SPEED, VERTICAL_SPEED
)

# Streamlit 页面配置
st.set_page_config(page_title="智能钢板堆垛系统", page_icon="⚙", layout="wide")

# 创建用于保存图像的目录
output_dir_base = "result/"
os.makedirs(output_dir_base, exist_ok=True)

st.title("⚙ 智能钢板堆垛系统")

# 使用 display_icon_with_header 函数替换现有的图标和标题显示逻辑
display_icon_with_header("data/icon/icon01.jpg", "数据导入", font_size="24px", icon_size="20px")

# 数据来源选择框
col3, col4, col11 = st.columns([0.01, 0.45, 0.55])
with col3:
    st.image("data/icon/icon02.jpg", width=20)
with col4:
    data_choice = st.selectbox("数据来源", ["上传自定义数据集", "使用测试数据集"])
with col11:
    st.image("data/icon/img.png", width=20)

df = None
dataset_name = None
system_data_dir = "data/Steel_Data"

# 数据集导入逻辑
if data_choice == "上传自定义数据集":
    uploaded_file = st.file_uploader("上传钢板数据集 (CSV)", type=["csv"])
    if uploaded_file:
        dataset_name = uploaded_file.name.split('.')[0]
        df = pd.read_csv(uploaded_file)
        st.write("已上传的数据集：")
        st.write(df.head())

elif data_choice == "使用测试数据集":
    col5, col6, col12 = st.columns([0.01, 0.44, 0.55])
    with col5:
        st.image("data/icon/icon02.jpg", width=20)
    with col12:
        st.image("data/icon/img.png", width=20)
    with col6:
        available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]
        selected_dataset = st.selectbox("测试数据集", [""] + available_datasets)
        if selected_dataset:
            dataset_name = selected_dataset
            system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
            df = pd.read_csv(system_dataset_path)

# 添加勾选按钮和 Start Work 按钮放在一行
col7, col8 = st.columns([0.15, 0.85])
with col7:
    deep_optimization = st.checkbox("启用深度优化")
with col8:
    start_work = st.button("Start Work")

use_adaptive = True

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

    # 默认混合优化参数
    hybrid_params = {
        'num_particles': PSO_PARAMS['num_particles'],
        'num_positions': len(Dki),
        'w': PSO_PARAMS['inertia_weight'],
        'c1': PSO_PARAMS['cognitive_component'],
        'c2': PSO_PARAMS['social_component'],
        'max_iter_pso': PSO_PARAMS['max_iterations'],
        'initial_temperature': SA_PARAMS['initial_temperature'],
        'cooling_rate': SA_PARAMS['cooling_rate'],
        'min_temperature': SA_PARAMS['min_temperature'],
        'max_iterations_sa': SA_PARAMS['max_iterations'],
        'lambda_1': SA_PARAMS['lambda_1'],
        'lambda_2': SA_PARAMS['lambda_2'],
        'lambda_3': SA_PARAMS['lambda_3'],
        'lambda_4': SA_PARAMS['lambda_4'],
        'dataset_name': dataset_name,
        'objectives': objectives,
        'use_adaptive': True  # 自适应模拟退火
    }

    if start_work:
        # 运行混合优化器
        run_optimization(PSO_SA_Optimizer, hybrid_params, df, DEFAULT_AREA_POSITIONS, output_dir_base, "psosa")








