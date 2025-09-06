import tensorflow as tf
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可以浮现
tf.random.set_seed(42)
np.random.seed(42)

# 配置参数
DATA_DIR = './data'         # 数据集目录
TRAIN_FILE = 'train.csv'    # 训练集文件名
TEST_FILE = 'test.csv'      # 测试集文件名
MAX_VOCAB_SIZE = 20000      # 词汇表大小
MAX_SEQ_LEN = 50            # 序列最大长度
BATCH_SIZE=64               # 批次大小
EPOCHS = 20                 # 训练轮数
REPORT_DIR = './report'     # 训练相关资料
MODEL_DIR = './models'      # 模型保存目录

# Transformer参数配置
NUM_LAYERS = 4              # encoder和decoder层数
D_MODEL = 128               # 模型维度
NUM_HEADS = 4               # 多头注意力的头数
UNIT = 128                  # 前馈神经网络的隐藏层单元数
DROPOUT_RATE = 0.1          # Dropout比率

# 加载数据
def load_data(data_dir,filename):
    """加载中文和英文的CSV翻译数据"""
    file_path = os.path.join(data_dir,filename)
    df = pd.read_csv(
        file_path,
        header=None,
        names=['chinese','english']
    )
    return df['english'].tolist(),df['chinese'].tolist()

# 分词
def preprocess_data(en_texts,zh_texts,max_vocab_size,max_seq_len):
    """创建中英文分词器"""
    processed_en = en_texts
    processed_zh = [f"<start> {text} <end>" for text in zh_texts]

    en_tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token='<unk>',
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    en_tokenizer.fit_on_texts(processed_en) # 统计并确定对应的数字编码
    en_sequences = en_tokenizer.texts_to_sequences(en_texts) # 将文本按照编码转换为数字序列
    en_padded = pad_sequences(
        en_sequences,
        maxlen=max_seq_len,     # 序列长度
        padding='post',         # 填充方式，长度不足的时候填充
        truncating='post'       # 截断方式，长度超过的时候截断
    )

    # 中文分词器
    zh_tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token='unk',
        filters=''
    )
    zh_tokenizer.fit_on_texts(processed_zh)
    zh_sequences = zh_tokenizer.texts_to_sequences(processed_zh)
    zh_padded = pad_sequences(
        zh_sequences,
        maxlen=max_seq_len,
        padding='post',
        truncating='post'
    )

    # 添加特殊词汇
    en_tokenizer.word_index['<pad>'] = 0
    en_tokenizer.index_word[0] = '<pad>'
    zh_tokenizer.word_index['<pad>'] = 0
    zh_tokenizer.index_word[0] = '<pad>'

    # 计算实际词汇表大小
    en_vocab_size = min(len(en_tokenizer.word_index) + 1, max_vocab_size)
    zh_vocab_size = min(len(zh_tokenizer.word_index) + 1, max_vocab_size)

    return (en_padded, zh_padded,
            en_tokenizer, zh_tokenizer,
            en_vocab_size, zh_vocab_size)

# 位置编码,继承自Layer
class PositionEncoding(Layer):
    def __init__(self,position,d_model):
        # positon:序列的最大长度 d_model:模型维度
        super(PositionEncoding,self).__init__()
        self.pos_encoding = self.positional_encoding(position,d_model)

    def get_angles(self,position,i,d_model):
        # 来自论文里面的公式，计算角度，奇数和偶数时是同一组角度公式
        position = tf.cast(position, tf.float32)  # 关键修改：将position转为浮点型
        i = tf.cast(i, tf.float32)  # 将i转为浮点型
        d_model = tf.cast(d_model, tf.float32)

        angles = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / d_model)
        return position * angles  # 现在两者都是float32，可以相乘

    def positional_encoding(self,position,d_model):
        angle_rads = self.get_angles(
            position=tf.range(position)[:,tf.newaxis],
            i=tf.range(d_model)[tf.newaxis,:],
            d_model=d_model
        )
        # 增加这一行，确保angle_rads是三维张量
        # angle_rads = tf.expand_dims(angle_rads, axis=-1)  # 形状变为 (position, d_model, 1)

        # 正确的切片方式，只在第二个维度上切片
        sines = tf.math.sin(angle_rads[:, 0::2])    # 偶
        cosines = tf.math.cos(angle_rads[:, 1::2])  # 奇

        pos_encoding = tf.concat([sines,cosines],axis=-1)
        pos_encoding = pos_encoding[tf.newaxis,...]
        return tf.cast(pos_encoding,tf.float32)

    def call(self,inputs):
        return inputs + self.pos_encoding[:,:tf.shape(inputs)[1],:]

# 掩码函数
def create_padding_mask(seq):
    # padding mask,屏蔽0的无意义填充值
    seq = tf.cast(tf.math.equal(seq,0),tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    # 防止解码器训练时偷看未来的单词，前瞻掩码
    # 得到上三角阵
    mask = 1-tf.linalg.band_part(tf.ones((size,size)),-1,0)
    return mask # (seq_len, seq_len)

def create_masks(inp,tar):
    # 编码器填充掩码
    enc_padding_mask = create_padding_mask(inp)

    # 解码器第二个注意力块用
    dec_padding_mask = create_padding_mask(inp)

    # 解码器第一个注意力块用
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask= create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask,look_ahead_mask)

    return enc_padding_mask,combined_mask,dec_padding_mask

# Transformer编码层
def encoder_layer(units,d_model,num_heads,dropout,name="encoder layer"):
    inputs = Input(shape=(None,d_model),name="inputs")

    # 多头注意力
    attention = MultiHeadAttention(
        key_dim = d_model // num_heads,
        num_heads = num_heads,
        name="attention"
    )(query=inputs, value=inputs, key=inputs)
    attention = Dropout(rate=dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    # 前馈网络
    outputs = Dense(units=units,activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return Model(inputs=inputs,outputs=outputs,name=name)

# Transformer编码器
def encoder(vocab_size,num_layers,units,d_model,num_heads,dropout,name="encoder"):
    inputs = Input(shape=(None,),name="inputs")

    # 嵌入层
    embedding = Embedding(vocab_size,d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model,tf.float32))

    # 位置编码
    embedding = PositionEncoding(vocab_size,d_model)(embedding)

    outputs = Dropout(rate=dropout)(embedding)

    # 堆叠多个编码器层
    for i in range(num_layers):
        outputs = EncoderLayer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"encoder_layer_{i}"
        )(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


# Transformer解码器层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        # 初始化所有子层，包括注意力、Dropout、归一化和前馈网络
        self.attention = MultiHeadAttention(key_dim=d_model // num_heads, num_heads=num_heads, name="attention")
        self.dropout1 = Dropout(rate=dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(units=units, activation='relu'),
            Dense(units=d_model)
        ], name="ffn")
        self.dropout2 = Dropout(rate=dropout)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # 注意力子层：输入作为query、key和value
        attn_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        # 添加残差连接和层归一化
        out1 = self.norm1(inputs + attn_output)

        # 前馈网络子层
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 添加残差连接和层归一化
        out2 = self.norm2(out1 + ffn_output)

        return out2

# Transformer解码器
# Transformer解码器
def decoder(vocab_size,num_layers,units,d_model,num_heads,dropout,name="decoder"):
    inputs = Input(shape=(None,),name="inputs")
    enc_outputs=Input(shape=(None,d_model),name="encoder_outputs")
    # New inputs for masks
    look_ahead_mask = Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")

    # 嵌入层
    embedding = Embedding(vocab_size,d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model,tf.float32))

    # 位置编码
    embedding = PositionEncoding(vocab_size,d_model)(embedding)

    outputs = Dropout(rate=dropout)(embedding)

    # 堆叠多个解码器层
    for i in range(num_layers):
        outputs = DecoderLayer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"decoder_layer_{i}"
        )(inputs=outputs, enc_outputs=enc_outputs, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

    return Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
# 新增：Transformer解码器层类
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.attention1 = MultiHeadAttention(key_dim=d_model // num_heads, num_heads=num_heads, name="attention_1")
        self.dropout1 = Dropout(rate=dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.attention2 = MultiHeadAttention(key_dim=d_model // num_heads, num_heads=num_heads, name="attention_2")
        self.dropout2 = Dropout(rate=dropout)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(units=units, activation="relu"),
            Dense(units=d_model)
        ], name="ffn")
        self.dropout3 = Dropout(rate=dropout)
        self.norm3 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask, training=False):
        # 第一个多头注意力（自注意力，带前瞻掩码）
        attn1_output = self.attention1(
            query=inputs, value=inputs, key=inputs, attention_mask=look_ahead_mask, training=training
        )
        attn1_output = self.dropout1(attn1_output, training=training)
        out1 = self.norm1(inputs + attn1_output)

        # 第二个多头注意力（编码器-解码器注意力）
        attn2_output = self.attention2(
            query=out1, value=enc_outputs, key=enc_outputs, attention_mask=padding_mask, training=training
        )
        attn2_output = self.dropout2(attn2_output, training=training)
        out2 = self.norm2(attn2_output + out1)

        # 前馈网络
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.norm3(ffn_output + out2)

        return out3

# 完整的Transformer模型
def transformer(
        vocab_size_enc,vocab_size_dec,
        num_layers,units,d_model,num_heads,dropout,
        name="transformer"
):
    # 编码器输入(英文)
    inputs = Input(shape=(None,),name="inputs")
    # 解码器输入(中文)
    dec_inputs = Input(shape=(None,),name="dec_inputs")

    # 获取掩码
    _, combined_mask, padding_mask = create_masks(inputs, dec_inputs)

    # 编码器
    enc_outputs = encoder(
        vocab_size=vocab_size_enc,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )(inputs)

    # 解码器
    dec_outputs = decoder(
        vocab_size=vocab_size_dec,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )(inputs=[dec_inputs, enc_outputs, combined_mask, padding_mask])

    # 输出层
    outputs = Dense(units=vocab_size_dec,activation='softmax')(dec_outputs)

    return Model(inputs=[inputs,dec_inputs],outputs=outputs,name=name)

# 自定义学习调度器，先热身，后衰减
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model, warmup_steps=10000):
       super(CustomSchedule, self).__init__()

       self.d_model = d_model
       self.d_model = tf.cast(self.d_model, tf.float32)

       self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        # 修正：添加get_config方法
        return {
            "d_model": self.d_model.numpy(),
            "warmup_steps": self.warmup_steps
        }

def translate(sentence,transformer_model,en_tokenizer,zh_tokenizer,max_seq_len):
    # 预处理输入句子
    def preprocess_english(text):
        text = text.strip().lower()
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r' +', ' ', text)
        return text

    sentence = preprocess_english(sentence)
    sentence = [sentence]

    # 转换为序列
    input_sequence = en_tokenizer.texts_to_sequences(sentence)
    input_sequence = pad_sequences(input_sequence,maxlen=max_seq_len,padding='post',truncating='post')

    # 目标序列初始化为<start>标记
    start_token = zh_tokenizer.word_index["<start>"]
    end_token = zh_tokenizer.word_index["<end>"]
    output_sequence = tf.expand_dims([start_token], 0)

    # 生成翻译结果
    for _ in range(max_seq_len):
        # 预测下一个词
        predictions = transformer_model([input_sequence, output_sequence], training=False)
        predictions = predictions[:, -1:, :] # 只关注最后一个时间步

        # 选择概率最高的词
        predicted_id = tf.cast(tf.argmax(predictions,axis=-1),tf.int32)

        # 如果遇到<end>标记，结束生成
        if predicted_id == end_token:
            break

        # 将预测的词添加到目标序列
        output_sequence = tf.concat([output_sequence,predicted_id],axis=-1)

    # 解码生成的序列
    translation = zh_tokenizer.sequences_to_texts(output_sequence.numpy())[0]
    # 移除特殊标记
    translation = translation.replace("<start>","").replace("<end>","").replace("<pad>", "").strip()
    # 移除多余空格
    translation = re.sub(r' +', ' ', translation)

    return translation


# 保存模型摘要到文本文件
def save_model_summary(model, save_path):
    with open(save_path, 'w') as f:
        # 创建一个临时的打印函数来捕获输出
        def print_to_file(*args, **kwargs):
            print(*args, file=f, **kwargs)

        # 打印模型摘要
        model.summary(print_fn=print_to_file)


def plot_training_history(history, save_path):
    plt.figure(figsize=(14, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')  # 训练损失
    plt.plot(history.history['val_loss'], label='Validation Loss')  # 验证损失
    plt.title('Training and Validation Loss')  # 训练与验证损失
    plt.xlabel('Epoch')  # 轮次
    plt.ylabel('Loss')  # 损失值
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')  # 训练准确率
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # 验证准确率
    plt.title('Training and Validation Accuracy')  # 训练与验证准确率
    plt.xlabel('Epoch')  # 轮次
    plt.ylabel('Accuracy')  # 准确率
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 主函数
def main():
    # 1.加载数据
    print("正在加载训练数据...")
    train_en,train_zh = load_data(DATA_DIR,TRAIN_FILE)
    print(f'训练集加载完成，共{len(train_en)}条数据')

    print("加载测试数据...")
    test_en, test_zh = load_data(DATA_DIR, TEST_FILE)
    print(f"测试数据加载完成，共 {len(test_en)} 条")

    # 2.数据预处理，构建分词器
    print("预处理数据...")
    (train_en_seq, train_zh_seq,
     en_tokenizer, zh_tokenizer,
     en_vocab_size, zh_vocab_size) = preprocess_data(
        train_en, train_zh, MAX_VOCAB_SIZE, MAX_SEQ_LEN
    )

    (test_en_seq, test_zh_seq, _, _, _, _) = preprocess_data(
        test_en, test_zh, MAX_VOCAB_SIZE, MAX_SEQ_LEN,
    )

    print(f"英文词汇表大小: {en_vocab_size}")
    print(f"中文词汇表大小: {zh_vocab_size}")

    # 3. 准备训练目标数据
    # 目标输入是目标序列的前n-1个字符，目标输出是目标序列的后n-1个字符
    train_zh_input = train_zh_seq[:, :-1]
    train_zh_target = train_zh_seq[:, 1:]

    test_zh_input = test_zh_seq[:, :-1]
    test_zh_target = test_zh_seq[:, 1:]

    # 4. 构建模型
    print("构建Transformer模型...")
    transformer_model = transformer(
        vocab_size_enc=en_vocab_size,
        vocab_size_dec=zh_vocab_size,
        num_layers=NUM_LAYERS,
        units=UNIT,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT_RATE
    )

    # 5.定义损失函数和优化器
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none'
    )

    def loss_function(real,pred):
        mask = tf.math.logical_not(tf.math.equal(real,0))
        loss_ = loss_object(real,pred)

        mask = tf.cast(mask,dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    # 学习率调度
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    # 编译模型
    transformer_model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # 6. 准备回调函数
    checkpoint_path = "./checkpoints/transformer"
    os.makedirs(checkpoint_path, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, "best_model"),
            monitor='val_loss',
            save_best_only=True,
            period=2,# 每2次保存检查点一次
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=3,
        #     min_lr=1e-6,
        #     verbose=1
        # )
    ]

    # 7. 训练模型
    print("开始训练...")
    history = transformer_model.fit(
        x=[train_en_seq, train_zh_input],
        y=train_zh_target,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([test_en_seq, test_zh_input], test_zh_target),
        callbacks=callbacks
    )

    # 8. 绘制并保存训练过程图
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    plot_path = os.path.join(REPORT_DIR, 'training_history.png')
    plot_training_history(history, plot_path)
    print(f"训练过程图已保存至: {plot_path}")
    # 保存模型摘要
    model_summary_path = os.path.join(REPORT_DIR, 'model_summary.txt')
    save_model_summary(transformer_model, model_summary_path)
    print(f'模型已保存至: {model_summary_path}')

    # 9. 保存最终模型
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    final_model_path = os.path.join(MODEL_DIR, 'final_model')
    transformer_model.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    # 10. 保存分词器
    import pickle
    with open(os.path.join(MODEL_DIR, 'en_tokenizer.pkl'), 'wb') as f:
        pickle.dump(en_tokenizer, f)
    with open(os.path.join(MODEL_DIR, 'zh_tokenizer.pkl'), 'wb') as f:
        pickle.dump(zh_tokenizer, f)
    print("分词器已保存")

    # 11. 测试翻译效果
    print("\n测试翻译效果:")
    test_sentences = [
        "Hello, how are you?",
        "I like machine learning.",
        "What is your name?",
        "This is a transformer model.",
        "I want to learn Chinese."
    ]

    for sentence in test_sentences:
        translation = translate(sentence, transformer_model, en_tokenizer, zh_tokenizer, MAX_SEQ_LEN)
        print(f"英文: {sentence}")
        print(f"中文翻译: {translation}\n")

if __name__ == '__main__':
    main()