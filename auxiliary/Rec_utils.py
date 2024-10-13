import os
import base64
import pandas as pd
import plotly.io as pio
import streamlit as st
from datetime import datetime

def clear_csv(file_path):
    """清空CSV文件内容"""
    if os.path.exists(file_path):
        open(file_path, 'w').close()  # 打开并清空文件内容


def is_csv_empty(file_path):
    """检查CSV文件是否为空"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0  # 文件存在且大小为0时返回True


def clear_folder(folder_path):
    """清空图片文件夹"""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)  # 删除文件夹中的所有图片


# 创建文件夹（如果不存在）
def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def append_to_csv(data, file_path):
    """将识别结果追加到CSV文件中"""
    df = pd.DataFrame(data)  # 将数据转换为DataFrame格式
    # 如果文件存在，则追加数据；否则创建新文件并写入表头
    if os.path.exists(file_path):
        if is_csv_empty(file_path):
            df.to_csv(file_path, mode='a', header=True, index=False)  # 追加模式，不写入表头
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # 追加模式，不写入表头
    else:
        df.to_csv(file_path, index=False)  # 如果文件不存在，创建文件并写入表头
        df.to_csv(file_path, mode='a', header=False, index=False)  # 再次追加数据以防止错误


# 保存柱状图为 HTML 文件
def save_chart_to_file(fig, chart_name, save_dir):
    # 获取当前时间并格式化为时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将时间戳添加到文件名中
    file_name = chart_name
    file_name = f"{chart_name}_{timestamp}.html"
    file_path = os.path.join(save_dir, file_name)
    # file_path = os.path.join(save_dir, f"{chart_name}.html")
    # 使用 plotly 的保存功能保存图表为轻量级 html 文件（引用 CDN）
    pio.write_html(fig, file_path, include_plotlyjs='cdn')
    return file_path

# 加载指定目录中的所有历史图表
def load_historical_charts(save_dir):
    charts = {}
    for filename in os.listdir(save_dir):
        if filename.endswith(".html"):
            chart_name = filename.replace(".html", "")
            charts[chart_name] = os.path.join(save_dir, filename)
    return charts

def process_steel_code(input_str):
    # 去除字符串末尾的空格
    input_str = input_str.rstrip()

    # 找到最后一个空格
    last_space_idx = input_str.rfind(' ')

    # 分离字符串为两部分：空格前和空格后的内容
    if last_space_idx != -1:
        # 空格前的部分保持不变
        before_space = input_str[:last_space_idx + 1]
        # 空格后的部分进行替换操作
        after_space = input_str[last_space_idx + 1:]
        #print(f"as1:{(after_space)}")

        # 替换大小写的'O'为'0'，大小写的'X'为'*'
        before_space = before_space.replace('GBIT', 'GB/T')
        after_space = after_space.replace('O', '0').replace('o', '0')
        after_space = after_space.replace('X', '*').replace('x', '*')
        after_space = after_space.replace('I', '1').replace('|', '1')
        after_space = after_space.replace('Z', '2').replace('z', '2')
        #print(f"as2:{after_space}")
        # 返回替换后的完整字符串
        return before_space + after_space
    else:
        # # 如果没有空格，直接处理整个字符串
        # input_str = input_str.replace('O', '0').replace('o', '0')
        # input_str = input_str.replace('X', '*').replace('x', '*')
        return input_str

# 将图片转换为 base64 格式
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 自定义一个函数来创建带有图标和标题的组合,并自定义标题大小(默认24px)
def display_icon_with_header(icon_path, header_text, font_size="24px"):
    if os.path.exists(icon_path):
        # 将图片转换为 Base64 格式
        img_base64 = image_to_base64(icon_path)
        # 使用 HTML 和 CSS 实现图标和标题的对齐，同时支持自定义字体大小
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" width="40" style="margin-right: 10px;">
                <h3 style="margin: 0; font-size: {font_size};">{header_text}</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(f"<h3 style='font-size: {font_size};'>{header_text}</h3>", unsafe_allow_html=True)