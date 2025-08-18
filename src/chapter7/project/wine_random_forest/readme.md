# Project：红酒分类模型

## 项目简介
本项目基于红酒（Wine）数据集，使用随机森林（Random Forest）算法构建分类模型，实现对红酒类别的预测。通过输入红酒的 13 项理化特征，模型可输出该红酒所属类别（class_0、class_1、class_2）及对应概率，并将结果保存为 CSV 文件。

## 数据集

使用 scikit-learn 内置的load_wine数据集，包含：

- 特征：13 项红酒理化指标，包括酒精含量（alcohol）、苹果酸（malic_acid）、灰分（ash）、灰分碱度（alcalinity_of_ash）、镁含量（magnesium）、总酚（total_phenols）等
- 标签：3 种红酒类别（对应 class_0、class_1、class_2）
- 样本量：共 178 条数据，各类别样本分布均衡

## 文件结构
- `training.py`：模型训练脚本，包含数据加载、可视化、划分、标准化、模型训练及评估等功能
- `application.py`：模型预测脚本，读取输入数据并输出预测结果至 CSV 文件
- `model.pkl`：训练好的随机森林模型文件（由training.py生成）
- `scaler.pkl`：特征标准化器（由training.py生成，用于数据预处理）
- `report.txt`：模型评估报告（由training.py生成）
- `data.csv`：输入数据示例（需用户提供，包含 13 项特征）
- `predicted_results.csv`：预测结果输出文件（由application.py生成）

## 使用步骤
### 1. 模型训练（可选）
若需重新训练模型，运行`training.py`：

```bash
python training.py
```
执行后将：

- 展示数据集基本信息及特征相关性热图（可视化特征间关联）
- 划分训练集（70%）和测试集（30%），使用StandardScaler进行特征标准化
- 训练包含 100 棵树的随机森林模型
- 生成模型评估报告report.txt，并保存模型为model.pkl、标准化器为scaler.pkl
2. 模型预测
运行`application.py`进行红酒类别预测：

```
bash
python application.py
```
程序将：

- 读取默认输入文件data.csv（可通过参数指定其他文件，如predict_wine_classes("input.csv", "output.csv")）
- 检查输入数据是否包含 13 项必要特征
- 对数据进行标准化处理后，使用模型预测类别及概率
- 将结果保存至predicted_results.csv，包含原始特征、预测类别（result）及各类别概率（class_0_prob、class_1_prob、class_2_prob）
##模型评估结果
根据report.txt，模型在测试集（54 条数据）上的表现如下：

|类别|精确率（precision）|召回率（recall）|F1 分数（f1-score）|样本数（support）|
|--|--|--|--|--|
|class_0|1.00|1.00|1.00|19|
|class_1|1.00|1.00|1.00|21|
|class_2|1.00|1.00|1.00|14|

- 总体准确率（accuracy）：1.00
- 宏平均（macro avg）：1.00（精确率、召回率、F1 分数）
- 加权平均（weighted avg）：1.00（精确率、召回率、F1 分数）

模型对三种红酒类别的分类效果优异，各项评估指标均达到 100%。
## 可能的错误及解决方法
- FileNotFoundError：提示找不到model.pkl、scaler.pkl或输入 CSV 文件，需先运行training.py生成模型和标准化器，或检查输入文件路径是否正确
- ValueError：输入数据缺少必要特征，需确保 CSV 文件包含所有 13 项特征（特征名需与required_features列表一致）
- 其他错误：将显示具体错误信息，可检查输入数据格式（如是否包含非数值型数据）或环境配置是否正确

