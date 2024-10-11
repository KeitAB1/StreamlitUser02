import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import time
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
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
# 设置 Streamlit 页面配置
st.set_page_config(page_title="Steel Plate Stacking Optimization", page_icon="⚙")

# 添加用户身份选择
user_type = st.radio("Select User Type", ["Admin", "User"], index=0)

# 显示不同的欢迎信息
if user_type == "Admin":
    st.title("Admin - Steel Plate Stacking Optimization")
else:
    st.title("User - Steel Plate Stacking Optimization")

# 根据用户身份控制页面访问
if user_type == "Admin":
    st.sidebar.subheader("Admin Pages")
    pages = {
        "StackingVisualization": "1_📹_StackingVisualization",
        "AlgorithmsComparison": "2_🔗_AlgorithmsComparison",
        "ImageRecognition": "3_💿️_ImageRecognition",
        "SteelPlateQ&A": "4_🤖_SteelPlateQ&A",
        "Introduction": "5_📰_Introduction"
    }
else:
    st.sidebar.subheader("User Pages")
    pages = {
        "StackingVisualization": "1_📹_StackingVisualization",
        "Introduction": "5_📰_Introduction"
    }

# 根据用户身份动态显示侧边栏选项
page_selection = st.sidebar.selectbox(
    "Select a page to access:",
    list(pages.values())
)
