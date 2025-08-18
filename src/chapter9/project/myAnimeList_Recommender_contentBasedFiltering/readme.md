# Project:基于content-based Filtering的动漫推荐系统

## 项目概述
这是一个基于深度学习和Content-based Filtering(CBF)的动漫推荐系统，能够根据用户特征和动漫特征预测用户对动漫的评分，并为特定用户生成个性化的动漫推荐。系统采用双塔神经网络结构，分别处理用户特征和动漫特征，通过计算特征向量的点积实现评分预测。

## 数据集说明
项目使用的数据集来自 [Kaggle](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews, "Anime Dataset with Reviews - MyAnimeList")，包含三个核心文件：

- `animes.csv`：动漫基本信息，包括动漫 ID、标题、类型、播出时间、集数、评分、评价人数等
- `profiles.csv`：用户资料信息，包括用户名、性别、生日、喜欢的动漫等
- `reviews.csv`：用户对动漫的评分数据，包括用户名、动漫 ID、评分等

数据集经过预处理后，提取了用户和动漫的关键特征，用于模型训练和推荐生成。

## 项目结构
```text
project/
├── code/                  # 代码目录
│   ├── Application.py     # 推荐应用程序（交互式推荐和评分预测）
│   ├── explore.py         # 数据探索与预处理脚本
│   └── model.py           # 模型训练脚本
├── dataset/               # 原始数据集目录
│   ├── animes.csv
│   ├── profiles.csv
│   └── reviews.csv
├── processed/             # 预处理后的数据目录（自动生成）
│   ├── animes.csv
│   ├── profiles.csv
│   └── reviews.csv
└── model/                 # 模型及相关文件目录（自动生成）
    ├── anime_recommender_regression_model.h5  # 训练好的模型
    ├── user_scaler.pkl    # 用户特征归一化器
    ├── anime_scaler.pkl   # 动漫特征归一化器
    └── report.txt         # 模型评估报告
```

## 使用方法
### 数据预处理
首先运行数据预处理脚本，处理原始数据并生成模型训练所需的特征：
```bash
cd code
python explore.py
```
预处理过程会：

- 从动漫数据中提取类型、季节、播出年份、集数等特征
- 从用户数据中提取星座、出生年份、性别、喜欢的类型等特征
- 清理评分数据，过滤无效评分
- 生成预处理后的文件到processed目录 
### 模型训练
使用预处理后的数据训练推荐模型：

```bash
python model.py
```
训练过程会：

- 加载预处理后的数据并划分训练集和测试集
- 构建并训练双塔神经网络模型
- 保存训练好的模型、特征归一化器到model目录
- 生成模型评估报告report.txt
### 生成推荐
运行应用程序使用训练好的模型进行评分预测或生成推荐：
```bash
python Application.py
```


程序提供两种功能：

- 预测特定动漫评分：输入用户名和动漫 ID，获取预测评分
- 推荐 Top N 动漫：输入用户名和推荐数量，获取个性化推荐列表
- 退出程序

## 模型说明
模型采用双塔神经网络结构：

- 用户塔（User Tower）：处理用户特征（包括出生年份、星座、性别、喜欢的动漫类型等），通过全连接层和 Dropout 层提取用户嵌入向量
- 动漫塔（Anime Tower）：处理动漫特征（包括集数、评分、流行度、类型、播出季节等），通过全连接层和 Dropout 层提取动漫嵌入向量


两个塔的输出向量通过点积计算相似度，再通过自定义激活函数将结果映射到 1-10 分的评分范围

## 模型评估：

测试集损失（MSE）：3.3537


平均绝对误差（MAE）：1.3285


实测评分预测还行，推荐效果一坨


## 注意事项
- 首次运行需确保dataset目录下存在完整的原始数据文件
- 数据预处理和模型训练需要一定时间，具体取决于硬件性能
- 若processed或model目录不存在，程序会自动创建
- 推荐结果的质量受模型性能和用户特征完整性影响
