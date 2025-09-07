# 英中翻译模型项目说明

## 项目概述
本项目基于Transformer模型实现英文到中文的翻译功能，包含数据预处理、模型训练（隐含）及交互式翻译应用。项目使用WMT英中翻译语料库作为训练数据，提供完整的从数据处理到模型部署的流程。但是作者训练出来的模型好像效果还是很差

## 数据集说明
使用的数据集来自Kaggle：[WMT Translation: Chinese-English](https://www.kaggle.com/datasets/cliheng/wmt-translation-chinese-english?select=wmt_corpus.csv)

该数据集包含大量中英文平行语料，适用于机器翻译模型的训练。

## 环境依赖
- Python 3.x
- pandas
- numpy
- tensorflow
- tqdm
- pickle
- argparse
- re

## 项目结构
```
.
├── dataset/              # 数据集目录
│   └── wmt_corpus.csv    # 原始数据集文件
├── data/                 # 预处理后的数据集
│   ├── train.csv         # 训练集
│   └── test.csv          # 测试集
├── models/               # 模型及分词器保存目录
│   ├── final_model/      # 训练好的模型
│   ├── en_tokenizer.pkl  # 英文分词器
│   └── zh_tokenizer.pkl  # 中文分词器
├── preProcessor.py       # 数据预处理脚本
├── model.py              # 数据预处理脚本
└── application.py        # 交互式翻译应用
```

## 数据预处理

### 功能说明
`preProcessor.py`用于对原始数据集进行清洗、过滤和划分，生成训练集和测试集。

主要处理步骤：
1. 分块读取大规模CSV数据
2. 去除空值数据
3. 过滤包含@符号的污染数据
4. 清洗中英文文本（保留有效字符，去除多余空格等）
5. 划分训练集（30,000条）和测试集（8,000条）
6. 保存处理后的数据集

### 使用方法
```bash
python preProcessor.py
```

配置参数可在脚本中修改：
- `INPUT_CSV`：输入CSV文件路径
- `OUTPUT_DIR`：输出目录路径
- `TRAIN_SIZE`：训练集大小
- `TEST_SIZE`：测试集大小
- `CHUNK_SIZE`：分块处理大小

## 训练模型

### 功能概述
`model.py` 实现了一个基于 Transformer 的英中翻译模型，包含数据加载、文本预处理、模型构建、训练及翻译测试等完整功能。通过该脚本可以完成从数据预处理后的数据加载到模型训练、评估及保存的全流程。

### 配置参数说明
在脚本开头定义了核心配置参数，可根据需求调整：

| 参数名称 | 含义 | 默认值 | 说明 |
|---------|------|-------|------|
| `DATA_DIR` | 数据集目录 | `'./data'` | 存放训练集（`train.csv`）和测试集（`test.csv`）的目录 |
| `TRAIN_FILE` | 训练集文件名 | `'train.csv'` | 训练数据文件名 |
| `TEST_FILE` | 测试集文件名 | `'test.csv'` | 测试数据文件名 |
| `MAX_VOCAB_SIZE` | 词汇表大小 | `32000` | 中英文分词器的最大词汇量 |
| `MAX_SEQ_LEN` | 序列最大长度 | `30` | 文本序列的最大长度，超过则截断，不足则填充 |
| `BATCH_SIZE` | 批次大小 | `128` | 训练时的批次样本数量 |
| `EPOCHS` | 训练轮数 | `300` | 模型训练的总轮数 |
| `REPORT_DIR` | 训练资料目录 | `'./report'` | 存放训练可视化结果和模型摘要的目录 |
| `MODEL_DIR` | 模型保存目录 | `'./models'` | 存放训练完成的模型和分词器的目录 |
| `PERIOD` | 检查点间隔 | `10` | 每多少轮保存一次模型检查点 |
| `WARMUP_STEPS` | 预热步数 | `10000` | 学习率调度器的预热步数 |
| `PATIENCE` | 早停耐心值 | `10` | 验证损失连续多少轮不下降则停止训练 |
| `NUM_LAYERS` | 编码器/解码器层数 | `2` | Transformer 中编码器和解码器的堆叠层数 |
| `D_MODEL` | 模型维度 | `64` | 模型的特征维度（嵌入维度） |
| `NUM_HEADS` | 多头注意力头数 | `4` | 多头注意力机制中的头数量 |
| `UNIT` | 前馈网络单元数 | `256` | 前馈神经网络隐藏层的单元数量 |
| `DROPOUT_RATE` | Dropout 比率 | `0.1` | 防止过拟合的 Dropout 概率 |

### 使用流程

#### 1. 前置准备
- 确保已通过 `preProcessor.py` 生成预处理后的训练集（`train.csv`）和测试集（`test.csv`），并存放于 `DATA_DIR` 目录下

#### 2. 模型训练
直接运行脚本即可启动训练流程：
```bash
python model.py
```
训练过程会自动执行以下步骤：
- 加载训练集和测试集数据
- 对文本进行分词、序列转换和填充
- 构建 Transformer 模型（包含编码器、解码器、注意力机制等组件）
- 配置学习率调度器（预热+衰减策略）和优化器
- 训练模型并通过早停策略防止过拟合
- 保存训练过程中的最佳模型

#### 3. 输出文件
训练完成后会生成以下文件：
- 模型文件：`./models/final_model`（完整模型）
- 分词器：`./models/en_tokenizer.pkl`（英文分词器）和 `./models/zh_tokenizer.pkl`（中文分词器）
- 训练报告：`./report/training_history.png`（损失和准确率曲线）和 `./report/model_summary.txt`（模型结构摘要）
- 检查点文件：`./checkpoints/transformer/best_model`（训练过程中性能最佳的模型）

#### 4. 翻译测试
脚本会自动对预设的测试句子进行翻译并打印结果，示例输出：
```
英文: Hello, how are you?
中文翻译: 你好，你好吗？

英文: I like machine learning.
中文翻译: 我喜欢机器学习。
```

### 自定义调整建议
- 若训练数据量较大，可增大 `BATCH_SIZE` 或减小 `MAX_SEQ_LEN` 以降低显存占用
- 若模型性能不足，可增加 `NUM_LAYERS`、`D_MODEL` 或 `NUM_HEADS` 提升模型容量（需注意显存限制）
- 若出现过拟合，可提高 `DROPOUT_RATE` 或减小模型规模
- 若训练不稳定，可调整 `WARMUP_STEPS` 或学习率相关参数

### 注意事项
- 首次运行会自动创建所需目录（`data`、`models`、`report`、`checkpoints`）
- 训练过程中会实时打印损失和准确率信息，可通过早停策略自动停止（默认耐心值为10）
- 模型支持断点续训，若中途停止，重新运行会加载最近的检查点继续训练
## 交互式翻译应用

### 功能说明
`application.py`提供交互式英文到中文的翻译功能，加载训练好的模型和分词器进行实时翻译。

### 使用方法
```bash
python application.py [--model-dir 模型目录] [--max-seq-len 最大序列长度]
```

参数说明：
- `--model-dir`：模型和分词器所在目录，默认值为`./models`
- `--max-seq-len`：序列最大长度，默认值为30

运行后，输入英文句子即可获得中文翻译，输入`exit`退出程序。

## 翻译流程
1. 对输入英文进行预处理（小写转换、标点处理等）
2. 将文本转换为整数序列
3. 序列填充/截断至指定长度
4. 使用Transformer模型进行预测生成
5. 将生成的序列解码为中文文本
6. 移除特殊标记并格式化输出结果

## 注意事项
1. 运行预处理脚本前，请确保已将数据集`wmt_corpus.csv`放置在`./dataset`目录下
2. 首次运行需先执行数据预处理生成训练集和测试集
3. 翻译应用依赖训练好的模型和分词器，请确保模型目录正确且文件完整