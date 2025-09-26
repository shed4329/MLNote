# Project: SE-CNN 消融实验项目说明文档
## 项目概述
本项目基于 CIFAR-100 图像分类数据集，通过**消融实验**验证 SE（Squeeze-and-Excitation）通道注意力模块的有效性。实验设计四组对照模型，聚焦 SE 模块核心组件（Squeeze 操作、激活函数）的作用，对比分析模型性能差异，揭示通道注意力机制的关键设计逻辑。


## 实验环境
### 硬件要求
- GPU：支持 CUDA 的 NVIDIA 显卡（推荐 6GB 及以上显存，如 RTX 4060/3060）
- CPU：多核处理器（如 Intel i5/i7 或 AMD Ryzen 5/7）
- 内存：16GB 及以上（避免数据加载时内存溢出）

### 软件依赖
- Python 3.8~3.10
- TensorFlow 2.8~2.10（含 CUDA、cuDNN 支持，需与 GPU 驱动匹配）
- 其他依赖库：
  ```bash
  pip install numpy matplotlib
  ```


## 实验设计
### 1. 数据集说明
- **数据集**：CIFAR-100（含 50000 张训练图、10000 张测试图，共 100 个类别，每张图尺寸 32×32×3）
- **数据预处理**：
  1. 归一化：像素值缩放至 [0,1]
  2. 标准化：基于训练集计算均值和标准差，对数据做 Z-Score 标准化
  3. 数据增强：随机旋转（±15°）、平移（±10%）、水平翻转、缩放（±10%）
  4. 标签编码：One-Hot 编码（适配多分类损失函数）

### 2. 模型设计（四组对照模型）
所有模型均基于 3 层卷积骨干网络，控制总参数量约 0.5M，仅差异在于是否引入 SE 模块及 SE 模块的结构：

| 模型编号 | 模型名称 | 核心差异 | 作用 |
|----------|----------|----------|------|
| 1 | plain CNN (without SE) | 无 SE 模块 | 基础对照组，验证 SE 模块的增益 |
| 2 | SE CNN without Squeeze | 移除 SE 模块的 Squeeze 操作（无全局平均池化） | 验证 Squeeze 操作对通道注意力的必要性 |
| 3 | SE CNN using ReLU | SE 模块的 Excitation 阶段用 ReLU 替代 Sigmoid | 验证激活函数对注意力权重生成的影响 |
| 4 | standard SE-CNN | 标准 SE 模块（Squeeze+Excitation+Sigmoid） | 实验组，SE 模块的完整实现 |

#### SE 模块核心逻辑
- **Squeeze**：全局平均池化（GlobalAveragePooling2D），将空间维度压缩为 1×1，保留通道全局信息
- **Excitation**：全连接层降维→激活→升维，生成通道注意力权重
- **Scale**：注意力权重与原特征图逐元素相乘，强化重要通道


### 3. 训练配置
- **优化器**：SGD（随机梯度下降）
  - 初始学习率：0.03
  - 动量（Momentum）：0.9（加速收敛，减少震荡）
- **正则化**：
  - 权重衰减（Weight Decay）：5e-4（L2 正则化，防止过拟合）
  - Dropout：0.5（全连接层后，抑制过拟合）
- **训练控制**：
  - 批次大小（Batch Size）：2048（平衡训练速度与内存占用）
  - 最大轮次（Epochs）：100
  - 早停机制（Early Stopping）：监控验证准确率，连续 10 轮无提升则停止并恢复最优权重
  - 学习率衰减（ReduceLROnPlateau）：监控验证损失，连续 5 轮无下降则学习率×0.1（最小降至 1e-6）


## 代码结构
```
SECNN_ablation_experiment/
├── model.py               # 主程序（模型定义、训练、结果生成）
└── experiment_results/    # 实验结果输出目录（自动生成）
    ├── experiment_report.txt  # 文本报告（准确率、训练时间、参数量）
    └── training_metrics.png   # 可视化图表（训练/验证损失、验证准确率对比）
```


## 快速开始
### 1. 环境准备
确保已安装对应版本的 Python 和 TensorFlow，建议使用虚拟环境隔离依赖：
```bash
# 创建虚拟环境
python -m venv .venv310
# 激活虚拟环境（Windows）
.venv310\Scripts\activate
# 安装依赖
pip install tensorflow==2.10 numpy matplotlib
```

### 2. 运行实验
直接执行主程序 `model.py`：
```bash
python src/chapter11/SECNN_ablation_experiment/model.py
```
- 程序启动后会自动检查 GPU，优先使用 GPU 训练（若无可使用 CPU）
- 训练过程中会打印每轮的损失、准确率，以及各模型的训练完成信息

### 3. 查看结果
实验结束后，结果会自动保存至 `experiment_results` 目录：
- **文本报告（experiment_report.txt）**：包含四组模型的测试准确率、训练时间、参数量，以及详细的训练历史（每轮损失、准确率）
- **可视化图表（training_metrics.png）**：3 个子图分别展示「训练损失对比」「验证损失对比」「验证准确率对比」，可直观观察模型收敛趋势


## 预期结果
### 1. 性能排序
标准 SE-CNN（模型 4）> SE CNN using ReLU（模型 3）> plain CNN（模型 1）> SE CNN without Squeeze（模型 2）

### 2. 预期结论（基于预期结果）
1. **SE 模块有效**：模型 4 准确率高于模型 1，证明通道注意力能提升特征表达能力
2. **Squeeze 操作必要**：模型 2 准确率最低，说明全局平均池化（Squeeze）是 SE 模块提取通道全局信息的核心
3. **激活函数影响大**：模型 3 准确率低于模型 4，因为 ReLU 可能输出负值或零，无法有效生成「0-1 注意力权重」（Sigmoid 更适配权重归一化）


## 注意事项
1. **GPU 内存不足**：若出现 `OutOfMemoryError`，可减小 `BATCH_SIZE`（如改为 1024 或 512）
2. **训练时间过长**：若仅需验证代码逻辑，可临时将 `EPOCHS` 改为 10~20（快速查看是否报错）
3. **结果复现**：由于随机种子未固定，每次训练结果可能有微小波动（±0.5% 准确率），建议多次训练取平均值


## TODO
1. 调整 SE 模块的 `reduction` 参数（默认 16），验证通道压缩比对性能的影响
2. 替换优化器为 Adam（学习率设为 0.001），对比 SGD 与 Adam 在该任务上的表现
3. 增加更多消融组（如移除 Excitation 操作），进一步拆解 SE 模块的作用
4. 在更大数据集（如 ImageNet）或更深模型（如 ResNet）上验证 SE 模块的泛化能力