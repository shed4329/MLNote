# KTH-TIPS 纹理分类：基于 CLBP 和 SVM 的实现

本项目基于**Completed Local Binary Pattern (CLBP)** 特征提取方法和**支持向量机 (SVM)**，实现了 KTH-TIPS 数据集的纹理分类任务。


## 项目介绍

KTH-TIPS 是一个经典的纹理材质数据集，包含铝箔、面包、灯芯绒等 10 类常见材质。本项目通过以下步骤完成分类：
1. 提取每张图像的 CLBP 特征（包含中心、符号、幅值三个分量的纹理描述）；
2. 用 SVM 分类器训练并评估模型；
3. 生成详细的分类报告和混淆矩阵，保存训练好的模型。


## 环境依赖

- Python 3.6+
- 主要库：
  ```bash
    pip install -r requirements.txt
  ```


## 数据集准备

1. 下载 KTH-TIPS 数据集：[官网链接](https://www.csc.kth.se/cvap/databases/kth-tips/)
2. 解压后，将数据集目录结构整理为：
   ```
   KTH_TIPS/
   ├── aluminium_foil/
   ├── brown_bread/
   ├── corduroy/
   ├── cotton/
   ├── cracker/
   ├── linen/
   ├── orange_peel/
   ├── sandpaper/
   ├── sponge/
   └── styrofoam/
   ```


## 快速启动


1修改代码中 `DATA_ROOT` 为你的数据集路径（如 `"./KTH_TIPS"`）。

2运行主程序：
   ```bash
   python main.py
   ```


## 代码结构说明

- `clbp_feature`: 向量化实现的 CLBP 特征提取函数，包含三个分量（中心、符号、幅值）的直方图拼接。
- `load_kth_tips`: 加载并预处理 KTH-TIPS 数据集，自动遍历类别并提取特征。
- `generate_report`: 生成分类报告（精确率、召回率、F1 分数）和混淆矩阵热力图。
- `main`: 主流程（加载数据、划分数据集、训练 SVM、评估并保存结果）。


## 参数说明

### CLBP 特征参数
- `R`: 邻域半径（默认 `2`），控制纹理感受野大小。
- `P`: 邻域点数（默认 `8`），控制方向分辨率。
- 幅值阈值（代码中 `abs(diffs) >= 10`）：控制对纹理对比度的敏感度。

### 分类器参数
- `SVM` 核函数：默认 `rbf`（径向基函数）。
- `C`: 正则化系数（默认 `10`），平衡分类准确率和模型复杂度。
- `gamma`: 核函数系数（默认 `0.1`），控制核函数的局部影响范围。


## 结果说明

运行后，在 `./results` 目录下会生成：
- `classification_report.txt`: 文本格式的分类报告（包含精确率、召回率、F1 分数）。默认参数的准确率为78.19%
- `confusion_matrix.png`: 混淆矩阵热力图，直观展示类别间的误分类情况。
- `clbp_svm_model.pkl`: 训练好的 SVM 模型。
- `feature_scaler.pkl`: 特征标准化器（用于后续预测新样本）。


## 性能优化建议

- **特征参数调优**：调整 `R`、`P` 和幅值阈值，平衡纹理描述能力和计算效率。
- **分类器调优**：用 `GridSearchCV` 搜索 SVM 的 `C` 和 `gamma` 最优组合。
- **特征增强**：尝试分块 CLBP（保留空间信息）或融合其他纹理特征（如 LBP、GLCM）。
