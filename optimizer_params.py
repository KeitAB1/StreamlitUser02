# optimizer_params.py

# 模拟退火 (SA) 参数配置
SA_PARAMS = {
    'initial_temperature': 1000.0,
    'cooling_rate': 0.9,
    'min_temperature': 0.1,
    'max_iterations': 100,
    'lambda_1': 1.0,
    'lambda_2': 1.0,
    'lambda_3': 1.0,
    'lambda_4': 1.0,
    'use_adaptive': True
}

# 遗传算法 (GA) 参数配置
GA_PARAMS = {
    'population_size': 100,
    'mutation_rate': 0.01,
    'crossover_rate': 0.7,
    'max_generations': 100,
    'lambda_1': 1.0,
    'lambda_2': 1.0,
    'lambda_3': 1.0,
    'lambda_4': 1.0
}

# 粒子群优化 (PSO) 参数配置
PSO_PARAMS = {
    'num_particles': 30,
    'inertia_weight': 0.5,
    'cognitive_component': 1.5,
    'social_component': 1.5,
    'max_iterations': 100,
    'lambda_1': 1.0,
    'lambda_2': 1.0,
    'lambda_3': 1.0,
    'lambda_4': 1.0
}

# 其他算法参数可以继续在这里定义
