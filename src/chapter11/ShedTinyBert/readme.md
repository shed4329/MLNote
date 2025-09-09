# Project: tinyBert训练与下游情感分类

## 项目概述

本项目实现了一个Mini-BERT/tinyBert模型的中文情感分析工具，能够对中文文本进行情感倾向判断（正面/负面）。项目包含完整的模型预训练、微调、评估和推理流程，使用酒店评论数据集（ChnSentiCorp_htl_8k）进行微调，可直接应用于各类中文文本的情感分析场景。由于BERT规模较小，训练和微调计算资源要求较低。训练单张普通显卡（如4060 laptop）即可，微调在高性能CPU上也可完成。但是由于模型预训练批次有限，参数有限，效果效果也很差，下游任务效果也不太行。

## 技术栈

- 深度学习框架：TensorFlow / Keras Core
- 自然语言处理：Hugging Face Transformers
- 数据集处理：Hugging Face Datasets
- 可视化：Matplotlib
- 其他：NumPy, argparse

## 项目结构

```
.
├── fine_tuning.py       # 优化的模型微调与训练脚本（推荐使用）
├── model.py             # 基础模型训练脚本
├── application.py       # 情感分析应用（支持命令行与交互式）
├── mini_bert_zh_hk.keras # 预训练的Mini-BERT模型（需自行准备或训练）
├── mini_bert_sentiment_classifier.keras # 微调后的情感分类模型
├── training_curves.png  # 训练过程中的损失与准确率曲线
└── README.md            # 项目说明文档
```

## 环境要求

```
tensorflow>=2.10.0
keras-core>=0.1.7
transformers>=4.20.0
datasets>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

安装依赖：
```bash
pip install tensorflow keras-core transformers datasets numpy matplotlib
```

## 模型预训练详解

### 预训练模型结构

项目中使用的`mini_bert_zh_hk.keras`是一个简化版的BERT模型，包含以下核心组件：

1. **BertEmbedding层**：
   - 词嵌入（token_embeddings）：将输入的词ID转换为向量表示
   - 位置嵌入（position_embeddings）：编码词在序列中的位置信息
   - 段落嵌入（segment_embeddings）：区分不同段落（本项目中主要用于单句，值均为0）
   - 层归一化（LayerNormalization）和 dropout：稳定训练过程

2. **TransformerEncoder层**：
   - 多头自注意力机制（MultiHeadAttention）：捕获文本中的上下文依赖关系
   - 前馈神经网络（FFN）：对注意力输出进行非线性变换
   - 残差连接（Residual Connection）和层归一化：缓解深度网络训练难题

3. **池化层（pooled_output）**：
   - 将序列输出转换为固定长度的向量表示，用于后续分类任务

### 预训练模型参数说明
以下是该 `model.py` 文件中主要参数的整理表格：

| 类别         | 参数名称          | 定义值       | 含义说明                                                                 |
|--------------|-------------------|--------------|--------------------------------------------------------------------------|
| 配置参数     | `MAX_SEQ_LEN`     | 64           | 文本序列的最大长度，超过则截断，不足则填充                               |
| 配置参数     | `EPOCHS`          | 15           | 模型训练的轮次（微调任务通常使用较少轮次）                               |
| 配置参数     | `BATCH_SIZE`      | 32           | 每次训练迭代的样本数量                                                   |
| 配置参数     | `LEARNING_RATE`   | 2e-5         | 优化器的学习率（BERT类模型微调常用较小学习率）                           |
| 配置参数     | `TEST_SIZE`       | 1/6          | 测试集占总数据的比例（约16.7%），用于训练集和测试集按5:1比例划分         |
| 自定义层参数 | `BertEmbedding`   | -            | 包含子参数：<br>- `vocab_size`：词汇表大小<br>- `max_seq_len`：最大序列长度（对应`MAX_SEQ_LEN`）<br>- `d_model`：嵌入维度<br>- `type_vocab_size`：分段词汇表大小（默认2）<br>- `dropout_rate`：dropout比例（默认0.1） |
| 自定义层参数 | `TransformerEncoder` | -        | 包含子参数：<br>- `d_model`：模型维度<br>- `num_heads`：多头注意力头数<br>- `dff`：前馈网络隐藏层维度<br>- `dropout_rate`：dropout比例（默认0.1）<br>- `normalize_first`：是否先进行层归一化（默认True） |

### 预训练模型获取与训练建议

1. **获取方式**：
   - 可通过别人已训练好的`mini_bert_zh_hk.keras`文件直接加载（仓库上不会提供）
   - 如需自行训练，需实现基于大规模中文语料的预训练逻辑，主要包括：
     - 配置合适的训练参数（通常需要较大的batch_size和训练轮次）

2. **训练注意事项**：
   - 由于训练数据量还是有点大，推荐使用GPU，在4060 laptop 上训练150epochs花费了大约3h

## 模型训练与微调详解

### 训练流程

1. **数据准备**
   - 自动从hugging face加载ChnSentiCorp_htl_8k酒店评论数据集
   - 按5:1比例划分训练集和测试集（可通过`TEST_SIZE`参数调整）
   - 文本预处理：分词、添加`[CLS]`和`[SEP]`标记、填充/截断至固定长度

2. **模型构建**
   - 加载预训练的Mini-BERT模型（`mini_bert_zh_hk.keras`）
   - 冻结预训练层参数（仅训练新增的分类层）
   - 添加分类输出层（1个神经元，sigmoid激活函数）

3. **执行训练**
   ```bash
   # 使用优化版本（推荐）
   python fine_tuning.py
   
   # 或基础版本
   python model.py
   ```

### 关键参数说明

| 参数名称 | 含义 | 默认值 | 作用 |
|---------|------|-------|------|
| `MAX_SEQ_LEN` | 文本最大长度 | 64 | 控制输入文本的长度，过长截断，过短填充 |
| `EPOCHS` | 训练轮次 | fine_tuning.py中为5，model.py中为15 | 决定模型在训练集上的迭代次数 |
| `BATCH_SIZE` | 批次大小 | 32 | 每次迭代训练的样本数量，影响训练速度和内存占用 |
| `LEARNING_RATE` | 学习率 | fine_tuning.py中为5e-5，model.py中为2e-5 | 控制参数更新幅度，过小收敛慢，过大可能不收敛 |
| `TEST_SIZE` | 测试集比例 | 1/6 | 测试集占总数据的比例 |
| `SEED` | 随机种子 | 42（仅fine_tuning.py） | 确保实验结果可重复 |
| `PLOT_SAVE_PATH` | 训练曲线保存路径 | "training_curves.png" | 训练过程可视化图像的保存位置 |

### 微调原理

- **迁移学习**：利用预训练的Mini-BERT模型学到的语言知识，针对情感分析任务进行专门优化
- **冻结策略**：代码中通过`layer.trainable = False`冻结了所有预训练层，仅训练最后添加的分类层（65个可训练参数）
- **优势**：训练效率高，避免过拟合，适合小数据集场景

### 训练输出说明

训练过程中会输出以下信息：
- 每轮训练的损失（loss）和准确率（binary_accuracy）
- 验证集的损失（val_loss）和准确率（val_binary_accuracy）
- 训练结束后会生成训练曲线图像（loss和accuracy变化趋势）
- 最终测试集评估结果（Test set loss和Test set accuracy）

示例输出：
```
40/40 ━━━━━━━━━━━━━━━━━━━━ 77s 2s/step - loss: 6.8275 - binary_accuracy: 0.6643
...
Test set loss: 0.6132
Test set accuracy: 0.6903
Fine-tuning completed!
Fine-tuned model saved as 'mini_bert_sentiment_classifier.keras'.
```

## 模型架构

微调后的模型架构如下：

```
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃     Param # ┃ Connected to                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_ids (InputLayer)        │ (None, 64)                │           0 │ -                              │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ token_type_ids (InputLayer)   │ (None, 64)                │           0 │ -                              │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ bert_embedding                │ (None, 64, 64)            │   1,356,544 │ input_ids[0][0],               │
│ (BertEmbedding)               │                           │             │ token_type_ids[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ transformer_encoder           │ (None, 64, 64)            │      49,984 │ bert_embedding[0][0]           │
│ (TransformerEncoder)          │                           │             │                                │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ transformer_encoder_1         │ (None, 64, 64)            │      49,984 │ transformer_encoder[0][0]      │
│ (TransformerEncoder)          │                           │             │                                │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ pooled_output (Lambda)        │ (None, 64)                │           0 │ transformer_encoder_1[0][0]    │
├───────────────────────────────┼───────────────────────────┼─────────────┼────────────────────────────────┤
│ classification_output (Dense) │ (None, 1)                 │          65 │ pooled_output[0][0]            │
└───────────────────────────────┴───────────────────────────┴─────────────┴────────────────────────────────┘
 Total params: 1,456,577 (5.56 MB)
 Trainable params: 65 (260.00 B)
 Non-trainable params: 1,456,512 (5.56 MB)
```

- **输入层**：接收`input_ids`（文本序列）和`token_type_ids`（段落标记）
- **BERT嵌入层**：将输入转换为包含词信息、位置信息和段落信息的向量
- **Transformer编码器**：通过自注意力机制捕获文本上下文关系
- **池化层**：将序列输出转换为固定长度的向量表示
- **分类层**：输出0-1之间的概率值（越接近1表示越可能是正面情感）

## 情感分析应用

### 使用方式

训练完成后，可通过`application.py`进行情感分析，支持两种模式：

#### 1. 命令行模式

直接分析指定文本：
```bash
python application.py --text "这家酒店环境非常好，服务也很贴心，下次还会再来！"
```

#### 2. 交互式模式

运行应用程序进入交互式分析：
```bash
python application.py
```

在交互式模式中，输入要分析的文本，程序会返回情感倾向及置信度。输入`exit`退出程序。

### 关键参数

| 参数名称 | 含义 | 默认值 | 作用 |
|---------|------|-------|------|
| `MAX_SEQ_LEN` | 文本最大长度 | 64 | 需与训练时保持一致 |
| `MODEL_PATH` | 模型文件路径 | "mini_bert_sentiment_classifier.keras" | 微调后模型的保存位置 |
| `POSITIVE_THRESHOLD` | 正面情感阈值 | 0.7 | 大于等于此值判断为正面情感，否则为负面 |

### 情感判断逻辑

- 模型输出为0-1之间的概率值
- 当概率≥0.7时，判断为**正面情感**
- 当概率<0.7时，判断为**负面情感**
- 置信度计算：正面情感为输出概率，负面情感为1-输出概率

## 注意事项

1. 运行前请确保`mini_bert_zh_hk.keras`预训练模型文件存在于项目根目录
2. 首次运行会自动下载BERT分词器（`bert-base-chinese`）和数据集，需要网络连接
3. 模型训练需要一定的计算资源，建议在GPU环境下运行以提高速度
4. 自定义层（`BertEmbedding`、`TransformerEncoder`）已在代码中定义，确保加载模型时能正确识别
5. `fine_tuning.py`相比`model.py`增加了固定随机种子和训练曲线可视化功能，推荐使用

## 扩展建议

1. 调整`application.py`中的`POSITIVE_THRESHOLD`阈值，根据实际需求优化情感判断
2. 尝试解冻部分预训练层（修改`layer.trainable = True`）进行更充分的微调
3. 增加更多评估指标，如精确率、召回率、F1分数等
4. 尝试使用更大的预训练模型或调整模型结构以提高性能
5. 扩展支持中性情感分类，将输出层改为3个神经元并使用softmax激活函数
6. 若需自行预训练模型，可扩展实现MLM（掩码语言模型）和NSP（下一句预测）任务