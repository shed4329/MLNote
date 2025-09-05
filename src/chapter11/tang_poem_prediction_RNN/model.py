import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

# 配置参数
data_dir = './data' # 唐诗txt所在的地方
corpus_file = './corpus.txt' # 处理后的txt文件
seq_length = 5 # 序列长度，构建数据集的时候要用
step = 1 # 构建数据集时每次滑动的步长
embedding_dim = 512 # 词嵌入维度
RNN_units = 256 # RNN隐藏层单元数
learning_rate = 0.001 # 学习率
epochs = 20 # 训练轮数
batch_size = 64 # 批次大小
oov_token = "<UNK>" # 未知词标记
eop_token = "" # 诗歌结束标记

# --- 1. 数据预处理 ---
# 1.1 合并txt文件
print('正在合并txt文件并添加结束标记...')
all_txt = ''
# 如果处理后的文件不存在，则重新合并所有源文件
# if not os.path.exists(corpus_file):
with open(corpus_file, 'w', encoding='utf-8') as f:
    # 遍历唐诗txt
    if not os.path.exists(data_dir):
        print(f"警告：目录 '{data_dir}' 不存在，请创建该目录并放入文件。")
    else:
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as infile:
                    all_txt += infile.read()
                    all_txt += eop_token  # 在每首诗后添加结束标记
        f.write(all_txt)
# 如果处理后的文件已存在，则直接读取
# else:
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         all_txt = f.read()
print('合并完成')

# 1.2 创建字符到索引的映射
# 获取所有不重复的字符，创建词汇表，并添加未知词标记
vocab = sorted(list(set(all_txt)))
vocab.insert(0, oov_token)
vocab_size = len(vocab)

# 创建字符到索引的字典
char_to_idx = {char:idx for idx,char in enumerate(vocab)}
idx_to_char = {idx:char for idx,char in enumerate(vocab)}
oov_idx = char_to_idx[oov_token]

print('词汇表大小:', vocab_size)
print('部分词汇表:',vocab[:20])

# 1.3 构建数据集
input_sequences = []
target_sequences = []

for i in range(0,len(all_txt)-seq_length,step):
    # 输入序列
    input_seq = all_txt[i:i+seq_length]
    # 目标序列
    target_seq = all_txt[i+1:i+seq_length+1]

    # 转换为对应索引，处理未知字符
    input_sequences.append([char_to_idx.get(ch, oov_idx) for ch in input_seq])
    target_sequences.append([char_to_idx.get(ch, oov_idx) for ch in target_seq])

# 转换为Numpy数组
X = np.array(input_sequences)
Y = np.array(target_sequences)

print(f"\n数据集构建完成，共有 {len(X)} 个训练样本。")
print(f"每个样本的形状为: ({seq_length},)")
print("\n第一个训练样本示例:")
print("输入 (文本):", all_txt[0:seq_length])
print("输入 (索引):", X[0])
print("目标 (文本):", all_txt[1:seq_length+1])
print("目标 (索引):", Y[0])

# --- 2. 模型构建与训练 ---
# 检查是否有 GPU 可用
print("\n检查GPU是否可用...")
print("可用GPU设备: ", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("已找到GPU，将使用GPU进行训练。")
else:
    print("未找到GPU，将使用CPU进行训练。")

# 2.1 构建模型
print("正在构建模型...")
model = Sequential([
    # Embedding 层将字符索引转换为向量
    Embedding(vocab_size, embedding_dim, input_length=seq_length),
    # SimpleRNN 层是你的核心RNN单元
    SimpleRNN(RNN_units, return_sequences=True),
    # DNN，增强特征提取
    Dense(256, activation='relu'),
    # Dense 层将RNN的输出转换为词汇表大小的预测
    Dense(vocab_size, activation='softmax')
])

# 2.2 编译模型
print("正在编译模型...")
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 查看模型结构
model.summary()

# 3.训练模型
print("\n开始训练模型...")
history = model.fit(
    X,
    Y,
    epochs=epochs,
    batch_size=batch_size
)

# 4. 保存模型和训练过程
# 保存模型结构和训练历史到文件
with open('training_history.txt', 'w', encoding='utf-8') as f:
    # 写入模型结构
    f.write('模型结构:\n')
    # 捕获model.summary()的输出
    from io import StringIO

    buffer = StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    model_summary = buffer.getvalue()
    f.write(model_summary)

    # 写入训练历史
    f.write("\n训练历史记录:\n")
    # 检查是否有accuracy键（有时可能是acc）
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history[acc_key]), start=1):
        f.write(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {acc:.4f}\n")

model.save('simple_rnn_tang_poem_generator.h5')
print("\n模型训练完成并已保存。")
