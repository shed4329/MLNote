# 基于 XGBoost 与 SHAP 的房价预测模型解释

本项目聚焦于**用 SHAP（SHapley Additive exPlanations）解释 XGBoost 房价预测模型**，通过可视化手段清晰展示“每个特征如何影响房价预测”，让模型决策过程从“黑盒”变为可解释的“白盒”。


## 核心：SHAP 解释的价值
SHAP 是一种基于博弈论的模型解释方法，能定量计算**每个特征对单个样本、全体样本的影响程度与方向**，解决了“模型为何预测这个房价？哪些因素在推动/拉低房价？”的核心问题。



## 运行与结果

1. 安装依赖：
```bash
pip install numpy pandas matplotlib xgboost scikit-learn shap
```

2. 运行代码：
```bash
python your_script_name.py
```

3. 输出结果：
- 自动生成两张图：`shap_waterfall_single_sample.png`（单样本瀑布图）、`shap_summary_plot.png`（全体样本摘要图）。
- 控制台会打印数据基本信息、模型训练与解释的过程日志。


通过这两种图，能直观回答：“哪些因素决定了房价？特征值多高时会推高/拉低房价？模型对单个房子的预测为何是这个数？”，真正实现模型从“能预测”到“可解释”的跨越。