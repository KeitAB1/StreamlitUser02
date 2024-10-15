# deep_optimizer_runner.py

import streamlit as st
from optimizers.psosa_optimizer import PSO_SA_Optimizer
from optimizers.eda_optimizer import EDA_with_Batch

# 添加深度优化功能，比较多种优化算法的结果，并选择最佳方案
def run_deep_optimization(optimizers_params, df, area_positions, output_dir_base):
    st.write("### 深度优化模式已启用，开始多种优化算法的运行...")

    # 存储每种优化算法的结果
    results = []
    optimizer_classes = {
        "PSO_SA_Optimizer": PSO_SA_Optimizer,
        "EDA_with_Batch": EDA_with_Batch,
        # 在这里可以添加其他的优化算法类，例如 GA, DE 等
    }

    # 逐一运行每种优化算法
    for optimizer_name, optimizer_params in optimizers_params.items():
        optimizer_class = optimizer_classes.get(optimizer_name)
        if optimizer_class is None:
            st.warning(f"未找到 {optimizer_name} 的优化器类，跳过此优化器。")
            continue

        st.write(f"### 运行 {optimizer_name} ...")
        optimizer = optimizer_class(**optimizer_params)
        optimizer.optimize()

        # 保存和可视化结果
        output_file_plates_with_batch = save_and_visualize_results(
            optimizer, df, area_positions, output_dir_base, optimizer_name
        )
        generate_stacking_distribution_statistics(df, area_positions, output_dir_base, optimizer_name)

        # 存储每个优化器的最佳分数
        results.append({
            "optimizer_name": optimizer_name,
            "best_score": optimizer.best_score,
            "output_file": output_file_plates_with_batch
        })

    # 比较所有优化器的结果，选择最佳分数
    if results:
        best_result = min(results, key=lambda x: x["best_score"])
        st.write(f"### 最佳优化器为：{best_result['optimizer_name']}，最佳得分为：{best_result['best_score']:.2e}")

        # 分区展示堆垛俯视图
        st.write("### 库区堆垛俯视图 - 最佳结果")
        generate_stacking_heatmaps(df, area_positions)

        # 提供下载按钮
        add_download_button(best_result['output_file'], best_result['optimizer_name'])
    else:
        st.warning("未找到任何可用的优化结果。")


