import numpy as np
import pandas as pd
import os
import streamlit as st
from datetime import datetime
from optimizers.psosa_optimizer import PSO_SA_Optimizer
from optimizers.eda_optimizer import EDA_with_Batch  # EDA 优化算法
from optimizers.ga_optimizer import GA_with_Batch  # 引入 GA 优化算法
from optimizers.co_ea_optimizer import CoEA_with_Batch  # 引入 CoEA 优化算法
from utils import save_and_visualize_results, generate_stacking_distribution_statistics, generate_stacking_heatmaps, show_stacking_distribution_statistics, show_stacking_height_distribution_chart, add_download_button  # 确保正确导入

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
        print("### 运行多种优化算法...")

        # 记录优化开始时间
        start_time = datetime.now()

        optimizer_classes = {
            "PSO_SA_Optimizer": PSO_SA_Optimizer,
            "EDA_with_Batch": EDA_with_Batch,  # 假设已实现 EDA 算法
            "GA_with_Batch": GA_with_Batch,  # 新增 GA 算法
            "CoEA_with_Batch": CoEA_with_Batch  # 新增 CoEA 算法
        }

        # 逐个算法进行优化
        for optimizer_name, optimizer_params in self.algorithms_params.items():
            optimizer_class = optimizer_classes.get(optimizer_name)
            if not optimizer_class:
                print(f"未找到 {optimizer_name} 的优化器类，跳过此优化器。")
                continue

            print(f"### 正在运行 {optimizer_name} ...")
            optimizer = optimizer_class(**optimizer_params)
            optimizer.optimize()

            # 保存和可视化结果
            output_file_plates_with_batch = save_and_visualize_results(optimizer, self.df, self.area_positions,
                                                                       self.output_dir_base, optimizer_name)
            # 这里只记录数据，不展示
            generate_stacking_distribution_statistics(self.df, self.area_positions, self.output_dir_base, optimizer_name)

            # 存储每个优化器的最佳分数及其结果
            self.results.append({
                "optimizer_name": optimizer_name,
                "best_score": optimizer.best_score,
                "output_file": output_file_plates_with_batch
            })

        # 选择最佳的优化结果
        self.select_best_result()

        # 记录优化结束时间并计算总运行时间
        end_time = datetime.now()
        total_runtime = end_time - start_time
        print(f"### 总共运行时间为：{total_runtime}")

        # 在Streamlit界面显示总共运行时间
        st.write(f"### 总共运行时间为：{total_runtime}")

    def select_best_result(self):
        """
        比较所有优化器的结果，选择最佳结果进行后续处理
        """
        if not self.results:
            print("未找到任何可用的优化结果。")
            return

        # 找到最佳结果（评分最低的）
        best_result = min(self.results, key=lambda x: x["best_score"])
        print(f"### 最佳优化器为：{best_result['optimizer_name']}，最佳得分为：{best_result['best_score']:.2e}")

        # 只展示最佳优化算法的垛位分布统计表和堆垛高度分布
        result_df, all_positions, all_heights = generate_stacking_distribution_statistics(
            self.df, self.area_positions, self.output_dir_base, best_result['optimizer_name'])

        # 先展示垛位分布统计表
        show_stacking_distribution_statistics(result_df, best_result['optimizer_name'], self.output_dir_base)

        # 再展示堆垛高度分布
        show_stacking_height_distribution_chart(all_positions, all_heights, best_result['optimizer_name'])

        # 生成库区堆垛俯视图
        print("### 库区堆垛俯视图 - 最佳结果")
        generate_stacking_heatmaps(self.df, self.area_positions)

        # 提供下载按钮
        add_download_button(best_result['output_file'], best_result['optimizer_name'])
