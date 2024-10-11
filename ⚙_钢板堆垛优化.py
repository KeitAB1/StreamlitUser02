import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
from optimizers.sa_optimizer import SA_with_Batch
from utils import save_convergence_history, add_download_button, run_optimization

# 从 constants 文件中引入常量
from constants import (
    OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH,
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

# 数据集导入
col1, col2 = st.columns([0.03, 0.97])  # 调整宽度比例
with col1:
    st.image("data/icon/icon01.jpg", width=20)
with col2:
    st.subheader("数据导入")

col3, col4 = st.columns([0.01, 0.99])
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
    col5, col6 = st.columns([0.01, 0.99])
    with col5:
        st.image("data/icon/icon02.jpg", width=20)
    with col6:
        available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]
        selected_dataset = st.selectbox("选择系统数据集", [""] + available_datasets)
        if selected_dataset:
            dataset_name = selected_dataset
            system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
            df = pd.read_csv(system_dataset_path)
            st.write(f"已选择系统数据集：{selected_dataset}")
            st.write(df.head())
        else:
            st.warning("请选择系统数据集")

# 优化参数配置
initial_temperature = 1000.0
cooling_rate = 0.9
min_temperature = 0.1
max_iterations_sa = 2
lambda_1, lambda_2, lambda_3, lambda_4 = 1.0, 1.0, 1.0, 1.0
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

    # 选择优化算法
    col7, col8 = st.columns([0.03, 0.97])  # 调整宽度比例
    with col7:
        st.image("data/icon/icon01.jpg", width=20)
    with col8:
        st.subheader("选择优化算法")




    sa_params = {
        'initial_temperature': initial_temperature,
        'cooling_rate': cooling_rate,
        'min_temperature': min_temperature,
        'max_iterations': max_iterations_sa,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'lambda_3': lambda_3,
        'lambda_4': lambda_4,
        'num_positions': num_positions,
        'num_plates': num_plates,
        'dataset_name': dataset_name,
        'objectives': objectives,
        'use_adaptive': use_adaptive
    }

    if st.button("开始优化"):
        st.info("优化进行中，请稍候...")
        run_optimization(SA_with_Batch, sa_params, df, DEFAULT_AREA_POSITIONS, output_dir_base, "sa")
