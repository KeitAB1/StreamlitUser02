import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
from optimizers.sa_optimizer import SA_with_Batch
from utils import save_convergence_history, add_download_button

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

st.title("智能钢板堆垛系统")
st.write("欢迎使用智能仓储优化解决方案！")

# 数据集导入
st.subheader("数据导入")
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
    available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]
    selected_dataset = st.selectbox("选择系统数据集", [""] + available_datasets)
    if selected_dataset:
        dataset_name = selected_dataset
        system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
        df = pd.read_csv(system_dataset_path)
        st.write(f"已选择系统数据集：{selected_dataset}")
        st.write(df.head())
    else:
        st.warning("请选择系统数据集以继续。")

# 优化参数
initial_temperature = 1000.0
cooling_rate = 0.9
min_temperature = 0.1
max_iterations_sa = 2
use_adaptive = True

# 优化分析
if df is not None:
    output_dir = os.path.join(output_dir_base, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

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

    sa_params = {
        'initial_temperature': initial_temperature,
        'cooling_rate': cooling_rate,
        'min_temperature': min_temperature,
        'max_iterations': max_iterations_sa,
        'num_positions': num_positions,
        'num_plates': num_plates,
        'lambda_1': 1.0,
        'lambda_2': 1.0,
        'lambda_3': 1.0,
        'lambda_4': 1.0,
        'dataset_name': dataset_name,
        'objectives': objectives,
        'use_adaptive': use_adaptive
    }

    # 缓存优化结果
    if 'best_position' not in st.session_state:
        if st.button("开始优化"):
            st.info("优化进行中，请稍候...")
            sa_optimizer = SA_with_Batch(**sa_params)
            best_position, best_score = sa_optimizer.optimize()
            st.session_state['best_position'] = best_position
            st.session_state['best_score'] = best_score
            st.session_state['heights'] = np.zeros(num_positions)
            for plate_idx, pos in enumerate(best_position):
                st.session_state['heights'][pos] += plates[plate_idx, 2]
            save_convergence_history(sa_optimizer.convergence_data, "SA", dataset_name, output_dir)
            st.success(f"优化完成！最佳得分：{best_score}")

            # 保存结果到 CSV 文件
            result_file_path = os.path.join(output_dir, f'final_stack_distribution_plates_{dataset_name}.csv')
            df.to_csv(result_file_path, index=False)
            st.session_state['result_file_path'] = result_file_path

    # 显示优化结果
    if 'best_position' in st.session_state:
        best_position = st.session_state['best_position']
        best_score = st.session_state['best_score']
        heights = st.session_state['heights']

        st.subheader("优化结果")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("选择图表类型：")
            chart_type = st.selectbox("图表类型", ["组合图 (柱状图+折线图)", "柱状图", "折线图", "面积图"])

            # 绘制图表的动态调整
            def get_bar_width(num_positions):
                if num_positions <= 3:
                    return 0.3
                elif num_positions <= 6:
                    return 0.2
                else:
                    return 0.1

            bar_width = get_bar_width(num_positions)
            if chart_type == "柱状图":
                fig = go.Figure(go.Bar(
                    x=list(range(num_positions)),
                    y=heights,
                    width=[bar_width] * num_positions,
                    marker_color='lightblue'
                ))
                fig.update_layout(title="堆垛高度分布 - 柱状图")
            elif chart_type == "折线图":
                fig = px.line(
                    x=list(range(num_positions)),
                    y=heights,
                    markers=True,
                    title="堆垛高度分布 - 折线图"
                )
            elif chart_type == "组合图 (柱状图+折线图)":
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(num_positions)),
                    y=heights,
                    name='柱状图',
                    width=[bar_width] * num_positions,
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(num_positions)),
                    y=heights,
                    mode='lines+markers',
                    name='折线图'
                ))
                fig.update_layout(title="堆垛高度分布 - 组合图")
            elif chart_type == "面积图":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(num_positions)),
                    y=heights,
                    fill='tozeroy',
                    mode='lines+markers',
                    name='面积图'
                ))
                fig.update_layout(title="堆垛高度分布 - 面积图")


            # 显示图表
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("优化详情：")
            st.write(f"最佳得分：{best_score}")
            st.write(f"钢板数量：{num_plates}")
            st.write(f"堆垛位置数量：{num_positions}")
            st.write("每个位置的最终高度：")
            st.dataframe(pd.DataFrame(heights, columns=["高度"]).T)

        # 使用 add_download_button 函数显示下载按钮和文件前5行
        if 'result_file_path' in st.session_state:
            add_download_button(st.session_state['result_file_path'], "SA")
