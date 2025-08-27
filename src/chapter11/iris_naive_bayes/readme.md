# Iris Dataset Naive Bayes Classification - README

## 项目概述
本项目基于**鸢尾花（Iris）数据集**，使用**高斯朴素贝叶斯（GaussianNB）算法**构建分类模型，实现对鸢尾花品种（山鸢尾、变色鸢尾、维吉尼亚鸢尾）的预测。项目包含数据加载、数据集分割、模型训练、性能评估、报告生成与模型保存等完整流程，适用于机器学习入门学习与分类任务实践。


## 功能模块
项目代码按逻辑划分为以下核心模块，各模块功能与作用如下：

| 模块                | 核心功能                                                                 |
|---------------------|--------------------------------------------------------------------------|
| 目录创建            | 自动生成 `classification_reports` 文件夹，用于存储分类评估报告           |
| 数据加载与探索      | 加载鸢尾花数据集，输出特征名、类别名、数据集维度等基础信息               |
| 数据集分割          | 按 7:3 比例（训练集70%、测试集30%）分割数据，确保结果可复现（`random_state=42`） |
| 模型训练            | 初始化高斯朴素贝叶斯模型，使用训练集数据完成模型拟合                     |
| 模型评估            | 计算模型准确率、生成混淆矩阵、分类报告（包含精确率、召回率、F1分数）     |
| 报告保存            | 生成带**时间戳**的评估报告文件，保存至 `classification_reports` 目录    |
| 模型持久化          | 使用 `pickle` 保存训练好的模型，支持后续直接加载使用                     |


## 模型结果
 |             |precision    |recall  |f1-score   |support
|-------------|-------------|--------|-----------|--------
|setosa       |1.00      |1.00      |1.00        |19
|versicolor       |1.00      |0.92      |0.96        |13
|virginica       |0.93      |1.00      |0.96        |13
|accuracy         |          |        |0.98        |45
|macro avg       |0.98      |0.97     | 0.97        |45
|weighted avg     |  0.98     | 0.98   |   0.98        |45
## 快速开始
### 1. 运行代码
在终端或IDE中执行以下命令：
```bash
python iris_nb_classification.py
```

### 2. 输出说明
#### （1）控制台输出
运行后将在控制台打印以下信息：
- 数据集基础信息（特征名、类别名、数据维度）；
- 模型评估结果（准确率、混淆矩阵、分类报告）；
- 评估报告保存路径、模型保存提示。

**示例控制台输出片段**：
```
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Class names: ['setosa' 'versicolor' 'virginica']
Dataset shape: features=(150, 4), targets=(150,)

Evaluation results:
Accuracy: 1.0000

Confusion matrix:
[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]

...（省略分类报告）...

Report saved to: /xxx/classification_reports/iris_classification_report_20240520_153045.txt
Model saved as iris_naive_bayes_model.pkl
```

#### （2）生成文件
代码运行后将自动生成2类文件：
1. **评估报告文件**：路径为 `classification_reports/iris_classification_report_YYYYMMDD_HHMMSS.txt`，包含完整的评估信息（生成时间、数据集信息、准确率、混淆矩阵、分类报告）；
2. **模型文件**：当前目录下生成 `iris_naive_bayes_model.pkl`，用于后续加载模型进行预测。


## 关键说明
1. **数据集说明**：鸢尾花数据集包含150个样本，每个样本有4个特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度），分为3个类别（各50个样本），是机器学习分类任务的经典数据集；
2. **算法选择**：高斯朴素贝叶斯（GaussianNB）适用于连续型特征（如本项目的花部尺寸特征），假设特征服从高斯分布，计算效率高，适合小数据集；
3. **结果可复现**：`train_test_split` 中设置 `random_state=42`，确保每次运行时数据集分割结果一致，模型训练与评估结果可复现；
4. **报告时间戳**：评估报告文件名包含时间戳（`YYYYMMDD_HHMMSS`），避免多次运行时文件覆盖，便于版本管理。
