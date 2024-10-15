import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# 从 constants 文件中引入常量
from constants import (
    OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH,
    DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS,
    HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE,
    INBOUND_POINT, OUTBOUND_POINT, Dki
)

def extract_timestamp_from_filename(file_name):
    parts = file_name.split('_')
    if len(parts) >= 5:
        date_part = parts[-2]
        time_part = parts[-1].replace('.csv', '')
        return f"{date_part}_{time_part}"
    return file_name

def calculate_statistics(data, file_names):
    statistics = []
    for i, scores in enumerate(data):
        mean_value = np.mean(scores)
        std_dev = np.std(scores)
        max_value = np.max(scores)
        min_value = np.min(scores)
        value_range = max_value - min_value
        best_score = scores[-1]
        timestamp = extract_timestamp_from_filename(file_names[i])
        statistics.append({
            'Timestamp': timestamp,
            'Mean': f"{mean_value:.2e}",
            'Std Dev': f"{std_dev:.2e}",
            'Max': f"{max_value:.2e}",
            'Min': f"{min_value:.2e}",
            'Range': f"{value_range:.2e}",
            'Best Score': f"{best_score:.2e}"
        })
    return pd.DataFrame(statistics)

def save_convergence_history(convergence_data, algorithm, dataset_name, history_data_dir):
    os.makedirs(history_data_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file_name = f"convergence_data_{algorithm}_{dataset_name}_{current_time}.csv"
    history_file_path = os.path.join(history_data_dir, history_file_name)
    convergence_df = pd.DataFrame(convergence_data, columns=['Iteration', 'Best Score'])
    convergence_df.to_csv(history_file_path, index=False)
    return history_file_path

def save_performance_metrics(best_score, worst_score, best_improvement, total_improvement,
                             iterations, time_elapsed, convergence_data, stable_iterations,
                             dataset_name, algorithm_name):
    iterations = len(convergence_data)
    convergence_rate_value = (convergence_data[-1][1] - convergence_data[0][1]) / iterations
    relative_error_value = abs(best_score) / (abs(best_score) + 1e-6)
    average_improvement = total_improvement / iterations if iterations > 0 else 0

    metrics = {
        'Best Score': best_score,
        'Worst Score': worst_score,
        'Best Improvement': best_improvement,
        'Average Improvement': average_improvement,
        'Iterations': iterations,
        'Time (s)': time_elapsed,
        'Convergence Rate': convergence_rate_value,
        'Relative Error': relative_error_value,
        'Convergence Speed (Stable Iterations)': stable_iterations
    }

    dataset_folder = f"result/comparison_performance/{dataset_name.split('.')[0]}"
    os.makedirs(dataset_folder, exist_ok=True)
    file_name = f"comparison_performance_{algorithm_name}.csv"
    file_path = os.path.join(dataset_folder, file_name)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(file_path, index=False)

def save_and_visualize_results(optimizer, df, area_positions, output_dir_base, algorithm_name):
    final_positions_with_batch = optimizer.best_position
    final_x, final_y = [], []

    for i, position in enumerate(final_positions_with_batch):
        area = position
        x, y = area_positions[area][i % len(area_positions[area])]
        final_x.append(x)
        final_y.append(y)

    df['Final Area'] = final_positions_with_batch
    df['Final X'] = final_x
    df['Final Y'] = final_y

    final_stack_dir = os.path.join(output_dir_base, 'final_stack_distribution')
    os.makedirs(final_stack_dir, exist_ok=True)
    output_file_plates_with_batch = os.path.join(final_stack_dir, f'final_stack_distribution_{algorithm_name}.csv')
    df.to_csv(output_file_plates_with_batch, index=False)

    heights_dict = {}
    df['Stacking Start Height'] = 0.0
    df['Stacking Height'] = 0.0

    for i in range(len(df)):
        area = df.loc[i, 'Final Area']
        x = df.loc[i, 'Final X']
        y = df.loc[i, 'Final Y']
        key = (area, x, y)
        current_height = heights_dict.get(key, 0.0)
        df.loc[i, 'Stacking Start Height'] = current_height
        df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
        heights_dict[key] = df.loc[i, 'Stacking Height']

    final_stack_plates_dir = os.path.join(output_dir_base, 'final_stack_distribution_plates')
    os.makedirs(final_stack_plates_dir, exist_ok=True)
    output_file_plates_with_batch = os.path.join(final_stack_plates_dir,
                                                 f'final_stack_distribution_plates_{algorithm_name}.csv')
    df.to_csv(output_file_plates_with_batch, index=False)

    return output_file_plates_with_batch



# 展示垛位分布统计表
def show_stacking_distribution_statistics(result_df, algorithm_name, output_dir_base):
    # 先创建保存路径
    final_stack_height_dir = os.path.join(output_dir_base, 'final_stack_distribution_height')
    os.makedirs(final_stack_height_dir, exist_ok=True)
    final_stack_distribution_path = os.path.join(final_stack_height_dir,
                                                 f'final_stack_distribution_height_{algorithm_name}.csv')
    result_df.to_csv(final_stack_distribution_path, index=False)

    # 显示表格
    st.dataframe(result_df)

    return final_stack_distribution_path

# 绘制堆垛高度分布图
def show_stacking_height_distribution_chart(all_positions, all_heights, algorithm_name):
    # 使用 display_icon_with_header 函数替换部分的展示
    col3, col4, col11 = st.columns([0.01, 0.25, 0.55])
    with col3:
        st.image("data/icon/icon02.jpg", width=20)
    with col4:
        # 选择图表类型
        chart_type = st.selectbox("选择图表类型", ["组合图 (柱状图+折线图)", "柱状图", "折线图", "面积图"],
                                  key=f"chart_selectbox_{algorithm_name}")

    def get_bar_width(num_positions):
        if num_positions <= 3:
            return 0.3
        elif num_positions <= 6:
            return 0.2
        else:
            return 0.1

    bar_width = get_bar_width(len(all_positions))
    if chart_type == "柱状图":
        fig = go.Figure(go.Bar(
            x=all_positions,
            y=all_heights,
            width=[bar_width] * len(all_positions),
            marker_color='lightblue'
        ))
        fig.update_layout(title="堆垛高度分布 - 柱状图")
    elif chart_type == "折线图":
        fig = px.line(
            x=all_positions,
            y=all_heights,
            markers=True,
            title="堆垛高度分布 - 折线图"
        )
    elif chart_type == "组合图 (柱状图+折线图)":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=all_positions,
            y=all_heights,
            name='柱状图',
            width=[bar_width] * len(all_positions),
            marker_color='lightblue'
        ))
        fig.add_trace(go.Scatter(
            x=all_positions,
            y=all_heights,
            mode='lines+markers',
            name='折线图'
        ))
        fig.update_layout(title="堆垛高度分布 - 组合图")
    elif chart_type == "面积图":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=all_positions,
            y=all_heights,
            fill='tozeroy',
            mode='lines+markers',
            name='面积图'
        ))
        fig.update_layout(title="堆垛高度分布 - 面积图")

    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

# 修改 generate_stacking_distribution_statistics，将计算逻辑与展示逻辑分离
def generate_stacking_distribution_statistics(df, area_positions, output_dir_base, algorithm_name):
    height_dict = {}
    plate_count_dict = {}

    # 初始化每个区域的高度和钢板计数
    for area in area_positions.keys():
        for pos in area_positions[area]:
            height_dict[(area, pos[0], pos[1])] = 0.0
            plate_count_dict[(area, pos[0], pos[1])] = 0

    def is_valid_position(area, x, y):
        return (area in area_positions) and ((int(x), int(y)) in area_positions[area])

    for index, row in df.iterrows():
        area = row['Final Area']
        x = row['Final X']
        y = row['Final Y']
        stacking_height = row['Stacking Height']

        x = int(x)
        y = int(y)

        if is_valid_position(area, x, y):
            height_dict[(area, x, y)] = stacking_height
            plate_count_dict[(area, x, y)] += 1

    results = []
    all_positions = []
    all_heights = []

    for area, positions in area_positions.items():
        total_plates = 0
        heights = []

        for pos in positions:
            height = height_dict[(area, pos[0], pos[1])]
            heights.append(height)
            all_positions.append(f"{area}-{pos[0]}-{pos[1]}")
            all_heights.append(height)
            total_plates += plate_count_dict[(area, pos[0], pos[1])]

        average_height = np.mean(heights)
        result_entry = {'Area': area, 'Total Plates': total_plates, 'Average Height': average_height}
        for i, pos in enumerate(positions):
            result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

        results.append(result_entry)

    result_df = pd.DataFrame(results)

    return result_df, all_positions, all_heights



def add_download_button(file_path, algorithm_name):

    # 使用 display_icon_with_header 函数替换现有的图标和标题显示逻辑
    display_icon_with_header("data/icon/icon01.jpg", "堆垛分布详情", font_size="24px", icon_size="20px")
    with open(file_path, 'rb') as file:
        st.download_button(
            label=f"Download Result",
            data=file,
            file_name=f'final_stack_distribution_plates_{algorithm_name}.csv',
            mime='text/csv'
        )
    df_plates_with_batch = pd.read_csv(file_path)
    st.dataframe(df_plates_with_batch.head(5))


# 生成单个库区的堆垛俯视热力图
def generate_single_area_heatmap(df, area, positions, zmin, zmax, width=300, height=300):
    x_values = []
    y_values = []
    z_values = []

    # 初始化每个位置的堆垛高度为0
    height_matrix = {pos: 0.0 for pos in positions}

    # 填充堆垛高度到矩阵中
    for index, row in df.iterrows():
        if row['Final Area'] == area:
            x = int(row['Final X'])
            y = int(row['Final Y'])
            stacking_height = row['Stacking Height']
            height_matrix[(x, y)] = stacking_height

    # 获取 x, y, z 数据
    for (x, y) in positions:
        x_values.append(x)
        y_values.append(y)
        z_values.append(height_matrix[(x, y)])

    # 创建热力图，使用Blues色阶，并设置 zmin 和 zmax
    fig = go.Figure(data=go.Heatmap(
        x=x_values,
        y=y_values,
        z=z_values,
        zmin=zmin,  # 设置统一的最小值
        zmax=zmax,  # 设置统一的最大值
        colorscale='Blues',  # 使用浅蓝色为主的色阶
        showscale=False  # 不显示颜色条
    ))

    # 设置 Y 轴反转，使 (0, 0) 从左上角开始
    fig.update_layout(
        title=f"库区 {area} 堆垛俯视图",
        xaxis_title="X 轴",
        yaxis_title="Y 轴",
        yaxis_autorange='reversed',  # 反转 Y 轴，使 (0, 0) 从左上角开始
        showlegend=False,
        width=width,  # 设置图表宽度
        height=height  # 设置图表高度
    )

    return fig


# 生成单独的颜色条
def generate_colorbar(zmin, zmax):
    fig = go.Figure()

    # 通过 Scatter 生成颜色条
    fig.add_trace(go.Scatter(
        x=[None],  # 不绘制图形，仅用于生成颜色条
        y=[None],  # 不绘制图形，仅用于生成颜色条
        mode='markers',
        marker=dict(
            colorscale='Blues',  # 使用Blues色阶
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(
                title="堆垛高度",
                orientation="h",  # 水平放置颜色条
                x=0.1,  # 放置在中央
                xanchor="center",  # 中心对齐
                thickness=20,  # 颜色条的厚度
                lenmode="pixels",  # 控制颜色条长度为像素
                len=400  # 设置颜色条长度，400像素
            ),
        ),
        hoverinfo='none'  # 不显示任何悬停信息
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),  # 调整边距，只保留顶部空间
        height=100,  # 设置高度，确保只显示颜色条
        xaxis=dict(showticklabels=False),  # 不显示X轴的标签
        yaxis=dict(showticklabels=False),  # 不显示Y轴的标签
        xaxis_visible=False,  # 不显示X轴
        yaxis_visible=False   # 不显示Y轴
    )

    return fig


# 生成多个库区的堆垛俯视热力图并排列
def generate_stacking_heatmaps(df, area_positions):
    # 计算所有库区中堆垛高度的最小值和最大值，以便统一色阶
    all_heights = df['Stacking Height'].values
    zmin = np.min(all_heights)  # 最小堆垛高度
    zmax = np.max(all_heights)  # 最大堆垛高度

    # 显示单独的颜色条
    st.plotly_chart(generate_colorbar(zmin, zmax), use_container_width=True)

    # 定义两行三列的布局
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    # 只让颜色条显示在顶部，其他热力图不显示颜色条
    with row1_col1:
        st.plotly_chart(generate_single_area_heatmap(df, 0, area_positions[0], zmin, zmax, width=300, height=300), use_container_width=True)
    with row1_col2:
        st.plotly_chart(generate_single_area_heatmap(df, 1, area_positions[1], zmin, zmax, width=300, height=300), use_container_width=True)
    with row1_col3:
        st.plotly_chart(generate_single_area_heatmap(df, 2, area_positions[2], zmin, zmax, width=300, height=300), use_container_width=True)
    with row2_col1:
        st.plotly_chart(generate_single_area_heatmap(df, 3, area_positions[3], zmin, zmax, width=300, height=300), use_container_width=True)
    with row2_col2:
        st.plotly_chart(generate_single_area_heatmap(df, 4, area_positions[4], zmin, zmax, width=300, height=300), use_container_width=True)
    with row2_col3:
        st.plotly_chart(generate_single_area_heatmap(df, 5, area_positions[5], zmin, zmax, width=300, height=300), use_container_width=True)



# 运行优化并展示堆垛俯视图和分布
def run_optimization(optimizer_class, params, df, area_positions, output_dir_base, algorithm_name):
    optimizer = optimizer_class(**params)
    optimizer.optimize()
    output_file_plates_with_batch = save_and_visualize_results(optimizer, df, area_positions, output_dir_base,
                                                               algorithm_name)
    generate_stacking_distribution_statistics(df, area_positions, output_dir_base, algorithm_name)

    # 分区展示堆垛俯视图
    st.write("### 库区堆垛俯视图")
    generate_stacking_heatmaps(df, area_positions)

    add_download_button(output_file_plates_with_batch, algorithm_name)




def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 自定义一个函数来创建带有图标和标题的组合,并自定义标题大小(默认24px)
def display_icon_with_header(icon_path, header_text, font_size="24px", icon_size="20px"):
    if os.path.exists(icon_path):
        # 将图片转换为 Base64 格式
        img_base64 = image_to_base64(icon_path)
        # 使用 HTML 和 CSS 实现图标和标题的对齐
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" width="{icon_size}" style="margin-right: 10px;">
                <h3 style="margin: 0; font-size: {font_size};">{header_text}</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(f"<h3 style='font-size: {font_size};'>{header_text}</h3>", unsafe_allow_html=True)





