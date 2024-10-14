import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime
from optimizers.psosa_optimizer import PSO_SA_Optimizer
from optimizers.eda_optimizer import EDA_with_Batch  # 假设有 EDA 优化算法
from utils import save_and_visualize_results, add_download_button, generate_stacking_distribution_statistics, generate_stacking_heatmaps  # 确保正确导入

class OptimizerRunner:
    def __init__(self, algorithms_params, df, area_positions, output_dir_base):
        """
        初始化类，指定优化算法、数据集、库区分布和输出路径
        :param algorithms_params: 各种算法及其参数的字典
        :param df: 待优化的数据集
        :param area_positions: 库区的分布情况
        :param output_dir_base: 输出文件保存路径
        """
        self.algorithms_params = algorithms_params
        self.df = df
        self.area_positions = area_positions
        self.output_dir_base = output_dir_base
        self.results = []

    def run_optimization(self):
        """
        运行所有指定的优化算法，并根据评分选择最佳结果
        """
        st.write("### 运行多种优化算法...")

        optimizer_classes = {
            "PSO_SA_Optimizer": PSO_SA_Optimizer,
            "EDA_with_Batch": EDA_with_Batch,  # 假设已实现 EDA 算法
        }

        # 逐个算法进行优化
        for optimizer_name, optimizer_params in self.algorithms_params.items():
            optimizer_class = optimizer_classes.get(optimizer_name)
            if not optimizer_class:
                st.warning(f"未找到 {optimizer_name} 的优化器类，跳过此优化器。")
                continue

            st.write(f"### 正在运行 {optimizer_name} ...")
            optimizer = optimizer_class(**optimizer_params)
            optimizer.optimize()

            # 保存和可视化结果
            output_file_plates_with_batch = save_and_visualize_results(optimizer, self.df, self.area_positions,
                                                                       self.output_dir_base, optimizer_name)
            generate_stacking_distribution_statistics(self.df, self.area_positions, self.output_dir_base, optimizer_name)

            # 存储每个优化器的最佳分数及其结果
            self.results.append({
                "optimizer_name": optimizer_name,
                "best_score": optimizer.best_score,
                "output_file": output_file_plates_with_batch
            })

        # 选择最佳的优化结果
        self.select_best_result()

    def select_best_result(self):
        """
        比较所有优化器的结果，选择最佳结果进行后续处理
        """
        if not self.results:
            st.warning("未找到任何可用的优化结果。")
            return

        # 找到最佳结果（评分最低的）
        best_result = min(self.results, key=lambda x: x["best_score"])
        st.write(f"### 最佳优化器为：{best_result['optimizer_name']}，最佳得分为：{best_result['best_score']:.2e}")

        # 生成库区堆垛俯视图并提供下载按钮
        st.write("### 库区堆垛俯视图 - 最佳结果")
        generate_stacking_heatmaps(self.df, self.area_positions)
        add_download_button(best_result['output_file'], best_result['optimizer_name'])
