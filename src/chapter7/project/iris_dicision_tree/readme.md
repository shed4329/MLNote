# Project: 鸢尾花分类模型

## 项目介绍

本项目基于鸢尾花（Iris）数据集，使用决策树算法构建分类模型，实现对鸢尾花种类的预测。通过输入花萼长度、花萼宽度、花瓣长度和花瓣宽度 4 个特征，模型可输出该鸢尾花属于 setosa、versicolor、virginica 三个品种的预测结果及对应概率。

## 数据集

使用 scikit-learn 内置的load_iris数据集，包含：

- 特征：花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）、花瓣宽度（petal width），单位均为 cm
- 标签：3 种鸢尾花品种（setosa、versicolor、virginica），分别对应数字 0、1、2
- 样本量：共 150 条数据，每个品种各 50 条

## 文件
- `training.py`：模型训练脚本，包含数据加载、可视化、划分、模型训练及评估等功能
- `application.py`：模型预测脚本，接收用户输入的特征并输出预测结果
- `model.pkl`：训练好的模型文件（由training.py生成）
- `report.txt`：模型评估报告（由training.py生成）

## 使用步骤
### 1. 模型训练（可选）
若需重新训练模型，运行`training.py`：

```bash
python training.py
```
执行后将：

- 展示数据集基本信息及可视化箱线图（按特征和品种分布）
- 划分训练集（70%）和测试集（30%），使用分层抽样保持类别比例
- 训练最大深度为 3 的决策树模型，避免过拟合
- 生成模型评估报告report.txt并保存模型为model.pkl
### 2. 模型预测
运行`application.py`进行鸢尾花种类预测：

```bash
python application.py
```


按照提示依次输入 4 个特征的数值（单位：cm），程序将输出：

- 预测的鸢尾花种类
- 属于各品种的概率（百分比形式）

## 模型评估结果
模型在测试集（45 条数据）上的表现如下：

|品种	|精确率（precision）	|召回率（recall）	|F1 分数（f1-score）	|样本数（support）
|-------|------------------|------------------|------------------|------------------|
|setosa	|1.00	|1.00	|1.00	|15
|versicolor	|1.00	|0.93	|0.97	|15
|virginica	|0.94	|1.00	|0.97	|15

- 总体准确率（accuracy）：0.98
- 宏平均（macro avg）：0.98（精确率、召回率、F1 分数）
- 加权平均（weighted avg）：0.98（精确率、召回率、F1 分数）

模型对三种鸢尾花的分类效果良好，尤其是 setosa 品种的预测准确率达到 100%。
### 可能的错误及解决方法
- FileNotFoundError：提示未找到model.pkl文件，需先运行training.py生成模型
- ValueError：输入格式不正确，需确保输入为数字（整数或小数）
- 其他错误：将显示具体错误信息，可检查输入值是否合理或环境配置是否正确