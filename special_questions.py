# special_questions.py

# 定义不同类别的特殊问题
def handle_special_questions(user_question):
    question = user_question.lower()  # 将用户输入转为小写，便于匹配

    # 关于机器人的问题
    if any(keyword in question for keyword in ["你是谁", "介绍", "assistant", "你的名字", "你是做什么的"]):
        return "我是湘潭大学开发的智能钢板堆垛问答助手，为您提供智能问答服务。"

    # 关于湘潭大学的问题
    elif any(keyword in question for keyword in ["湘潭大学", "xiangtan university", "学校", "大学", "历史", "简介"]):
        return "湘潭大学是一所具有悠久历史的高等学府，位于中国湖南省。"

    # 关于系统功能的问题
    elif any(keyword in question for keyword in ["系统功能", "系统", "功能", "system function", "能做什么"]):
        return "本系统提供智能钢板堆垛优化、钢板编号识别、算法性能对比、数据可视化等多种功能。"

    # 关于堆垛优化的问题
    elif any(keyword in question for keyword in ["堆垛优化", "优化算法", "堆垛", "stack optimization", "钢板堆垛", "仓储优化"]):
        return "智能堆垛优化使用多种算法如PSO、SA等，基于钢板的尺寸和重量等属性生成最优堆垛方案。"

    # 关于钢板相关问题
    elif any(keyword in question for keyword in ["钢板", "steel plate", "板材", "厚度", "尺寸", "材料"]):
        return "钢板是建筑和工业中常用的材料，通常厚度在0.2毫米到50毫米不等。"

    # 关于算法的问题
    elif any(keyword in question for keyword in ["算法", "algorithm", "pso", "遗传算法", "genetic algorithm", "模拟退火", "优化方法"]):
        return "本系统采用多种优化算法，如粒子群算法（PSO）、遗传算法（GA）、模拟退火算法（SA）等，来提高钢板堆垛的效率。"

    # 关于仓储管理问题
    elif any(keyword in question for keyword in ["仓储管理", "仓库", "存储", "库存", "仓储", "管理"]):
        return "本系统可以帮助优化仓储中的钢板堆垛，提高仓储空间利用率，减少翻垛和搬运时间。"

    # 关于数据可视化的问题
    elif any(keyword in question for keyword in ["数据可视化", "图表", "可视化", "data visualization", "结果展示"]):
        return "系统支持多种数据可视化形式，包括堆垛图、折线图、柱状图等，帮助用户更好地理解优化结果。"

    # 关于系统的技术实现问题
    elif any(keyword in question for keyword in ["技术实现", "技术", "技术架构", "system architecture", "实现方式"]):
        return "本系统使用Python语言开发，前端使用Streamlit框架，结合百度百科API等技术实现智能问答。"

    # 关于大数据分析的问题
    elif any(keyword in question for keyword in ["大数据", "big data", "数据分析", "数据处理", "分析方法"]):
        return "系统可以对堆垛数据进行大数据分析，结合历史数据来优化未来的堆垛和仓储管理。"

    # 关于钢板堆垛系统的改进问题
    elif any(keyword in question for keyword in ["系统改进", "改进建议", "提高效率", "改进", "enhancements"]):
        return "我们计划通过引入更多的优化算法和大模型来进一步提高钢板堆垛系统的智能化水平。"

    # 关于节能问题
    elif any(keyword in question for keyword in ["节能", "能源", "energy saving", "能效", "能耗", "节约能源"]):
        return "智能钢板堆垛系统通过优化堆垛顺序和路径规划，减少翻垛和搬运时间，从而提高能效，节约能源。"

    # 关于3D可视化的问题
    elif any(keyword in question for keyword in ["3d可视化", "三维可视化", "3d visualization", "3d展示", "三维展示"]):
        return "系统支持3D可视化展示，帮助用户查看钢板堆垛的3D模型，动态调整堆垛方案。"

    # 关于系统使用方法的问题
    elif any(keyword in question for keyword in ["使用方法", "怎么用", "使用教程", "guide", "使用说明", "如何使用"]):
        return "用户可以通过系统的交互式界面输入问题，系统会根据优化算法和数据库提供最优的钢板堆垛方案。"

    # 关于钢板编号识别的问题
    elif any(keyword in question for keyword in ["钢板编号", "编号识别", "图像识别", "ocr", "识别"]):
        return "系统使用OCR技术识别钢板编号，结合图像处理技术自动录入钢板信息，提高识别效率。"

    # 关于未来计划的问题
    elif any(keyword in question for keyword in ["未来计划", "未来发展", "计划", "未来展望", "future plan"]):
        return "系统未来将引入更多的AI技术和大模型优化，实现更智能化的钢板堆垛和仓储管理。"

    # 关于系统的性能问题
    elif any(keyword in question for keyword in ["性能", "性能测试", "系统性能", "performance", "效率"]):
        return "本系统在性能优化上表现出色，能够处理大规模的堆垛数据，同时快速响应用户的需求。"

    # 关于翻垛和搬运的问题
    elif any(keyword in question for keyword in ["翻垛", "搬运", "堆垛顺序", "搬运路径", "handling", "翻动"]):
        return "系统通过优化堆垛顺序和搬运路径，减少不必要的翻垛动作，提升仓储的整体效率。"

    # 关于钢板的材料特性问题
    elif any(keyword in question for keyword in ["材料特性", "钢板特性", "钢板材料", "material properties", "硬度", "强度"]):
        return "钢板的主要特性包括抗拉强度、硬度、韧性和耐腐蚀性，这些特性在堆垛优化时会被考虑进去。"

    # 关于钢板运输问题
    elif any(keyword in question for keyword in ["运输", "物流", "钢板运输", "transportation", "物流优化"]):
        return "钢板的运输和物流可以通过车辆路径规划（VRP）算法进行优化，以降低运输成本并提高效率。"

    # 关于机器人自动化问题
    elif any(keyword in question for keyword in ["机器人", "自动化", "自动化设备", "robot", "automation", "机械手"]):
        return "未来的堆垛系统将集成更多的机器人自动化设备，实现堆垛、翻垛、搬运等流程的全自动化管理。"

    # 关于算法性能对比的问题
    elif any(keyword in question for keyword in ["算法对比", "性能对比", "算法", "algorithm comparison", "pso和ga对比"]):
        return "系统提供多种算法的性能对比图，包括收敛曲线、计算时间和优化效果，帮助用户选择最优算法。"

    # 关于系统使用场景的问题
    elif any(keyword in question for keyword in ["使用场景", "应用场景", "场景", "scenario", "应用"]):
        return "本系统适用于各类钢材制造和存储的场景，尤其是在需要高效堆垛和仓储管理的生产环境中。"

    # 用户问"在吗"
    elif any(keyword in question for keyword in ["在吗", "在"]):
        return "在呢。请问，有什么问题呢？"

    # 默认返回None，表示没有匹配的特殊问题
    return None
