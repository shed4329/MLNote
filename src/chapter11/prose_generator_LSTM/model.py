import os.path
import pickle

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# 配置参数
data_dir = './data'
seq_length = 256
step = 1
LSTM1_unit = 256
LSTM2_unit = 256
batch_size = 128
epochs = 150

def read_data(dir = data_dir):
    """
    读取数据
    :param dir:文件路径
    :return: 拼接文件
    """
    str = ''
    if os.path.exists(dir):
        for file in os.listdir(dir):
            if file.endswith('.txt'):
                with open(os.path.join(dir, file), 'r', encoding='utf-8') as f:
                    str += f.read()

    return str

def main():
    # 1.数据预处理
    data = read_data()
    with open('text.txt', 'w', encoding='utf-8') as f:
        f.write(data)
    # 将字符映射为数字
    unique_char = sorted(list(set(data)))
    char_to_idx = {char:idx for idx,char in enumerate(unique_char)}
    idx_to_char = {idx:char for idx,char in enumerate(unique_char)}

    n_chars = len(data)
    n_vocab = len(unique_char)
    print(f'语料库共有{n_chars}个字符，{n_vocab}个不同的字符')

    # 2.生成数据集
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, step):
        seq_in = data[i:i+seq_length]
        seq_out = data[i+seq_length]
        dataX.append([char_to_idx[char] for char in seq_in])
        dataY.append(char_to_idx[seq_out])

    n_patterns = len(dataX)

    # 转为Numpy数组
    X = np.reshape(dataX, (n_patterns, seq_length, step))
    # 数据归一化
    X = X / float(n_vocab)
    # 输出onehot编码
    y = tf.keras.utils.to_categorical(dataY)
    # y = np.array(dataY)

    # 3.定义模型
    model = Sequential()
    model.add(LSTM(LSTM1_unit, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(LSTM2_unit))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.summary()

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # 3. 训练模型
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # 4. 保存模型
    model.save('model.h5')
    print('模型已保存')
    pickle.dump(char_to_idx, open('char_to_idx.pkl', 'wb'))
    pickle.dump(idx_to_char, open('idx_to_char.pkl', 'wb'))

if __name__ == '__main__':
    main()
