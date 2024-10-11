# UI.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
# from optimization_algorithms import SA_with_Batch
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
from utils import save_convergence_history

app = Flask(__name__)

# 设置目录路径（可根据需要调整）
OUTPUT_DIR = "result/output"
CONVERGENCE_DIR = "result/convergence"
DATA_DIR = "data"

# 创建输出和数据目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


@app.route('/')
def home():
    return "Welcome to the Optimization API. Use the endpoints /upload-dataset and /run-optimization."


@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    dataset_name = file.filename.rsplit('.', 1)[0]
    file_path = os.path.join(DATA_DIR, file.filename)
    file.save(file_path)

    df = pd.read_csv(file_path)
    return jsonify({
        "message": "Dataset uploaded successfully",
        "dataset_name": dataset_name,
        "columns": df.columns.tolist(),
        "sample_data": df.head().to_dict(orient='records')
    })


@app.route('/run-optimization', methods=['POST'])
def run_optimization():
    try:
        data = request.json
        dataset_name = data['dataset_name']
        initial_temperature = data['initial_temperature']
        cooling_rate = data['cooling_rate']
        min_temperature = data['min_temperature']
        max_iterations = data['max_iterations']
        lambda_1 = data.get('lambda_1', 1.0)
        lambda_2 = data.get('lambda_2', 1.0)
        lambda_3 = data.get('lambda_3', 1.0)
        lambda_4 = data.get('lambda_4', 1.0)

        # 读取数据集
        df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}.csv"))
        plates = df[['Length', 'Width', 'Thickness', 'Material_Code', 'Batch', 'Entry Time', 'Delivery Time']].values
        heights = np.zeros(len(plates))
        df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
        df['Entry Time'] = pd.to_datetime(df['Entry Time'])
        delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values
        batches = df['Batch'].values

        # 设置优化目标
        objectives = OptimizationObjectives(
            plates=plates,
            heights=heights,
            delivery_times=delivery_times,
            batches=batches,
            Dki=[],  # 使用默认参数或其他配置
            area_positions=[],  # 使用默认参数或其他配置
            inbound_point=(0, 0),
            outbound_point=(10, 10),
            horizontal_speed=1.0,
            vertical_speed=1.0
        )

        # 设置 SA 优化器参数
        sa_params = {
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(heights),
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': data.get('use_adaptive', False)
        }

        # 运行 SA 优化
        sa_optimizer = SA_with_Batch(**sa_params)
        best_position, best_score = sa_optimizer.optimize()

        # 保存收敛数据
        save_convergence_history(sa_optimizer.convergence_data, "SA", dataset_name, CONVERGENCE_DIR)

        return jsonify({
            "message": "Optimization completed successfully",
            "best_position": best_position.tolist(),
            "best_score": best_score,
            "convergence_data": sa_optimizer.convergence_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
