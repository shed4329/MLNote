import numpy as np
import tensorflow as tf
import keras_core as keras
from transformers import BertTokenizer
from datasets import load_dataset
from keras_core import layers
from keras_core.saving import register_keras_serializable

# 配置参数
MAX_SEQ_LEN = 64
EPOCHS = 15  # 微调通常只需要较少的 epochs
BATCH_SIZE = 32
LEARNING_RATE = 2e-5  # 学习率
TEST_SIZE = 1/6  # 测试集占比（1:5划分）
# 忽略烦人的警告
import warnings
warnings.filterwarnings("ignore")

# 1. 导入和定义自定义层和函数，确保在模型加载时可见
@register_keras_serializable()
class BertEmbedding(layers.Layer):
    def __init__(self, vocab_size, max_seq_len, d_model, type_vocab_size=2, dropout_rate=0.1, **kwargs):
        super().__init__(** kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=d_model, name="token_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            input_dim=max_seq_len, output_dim=d_model, name="position_embeddings"
        )
        self.segment_embeddings = layers.Embedding(
            input_dim=type_vocab_size, output_dim=d_model, name="segment_embeddings"
        )
        self.dropout = layers.Dropout(rate=dropout_rate, name="dropout")
        self.layer_norm = layers.LayerNormalization(epsilon=1e-12, name="layer_norm")

    def call(self, inputs):
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        batch_size = keras.ops.shape(input_ids)[0]
        position_ids = keras.ops.arange(self.max_seq_len, dtype='int32')
        position_ids = keras.ops.tile(
            keras.ops.expand_dims(position_ids, axis=0),
            [batch_size, 1]
        )
        token_embed = self.token_embeddings(input_ids)
        position_embed = self.position_embeddings(position_ids)
        segment_embed = self.segment_embeddings(token_type_ids)
        embeddings = token_embed + position_embed + segment_embed
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

@register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, normalize_first=True, **kwargs):
        super().__init__(** kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.normalize_first = normalize_first
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        if self.normalize_first:
            attn_output = self.mha(self.layernorm1(x), x, x, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = x + attn_output
            ffn_output = self.ffn(self.layernorm2(out1))
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = out1 + ffn_output
        else:
            attn_output = self.mha(x, x, x, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)
        return out2

@register_keras_serializable()
def ignore_neg100_sparse_categorical_accuracy(y_true, y_pred):
    mask = tf.not_equal(y_true, -100)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.metrics.sparse_categorical_accuracy(y_true_masked, y_pred_masked)

def preprocess_function(examples, tokenizer, max_seq_len):
    """数据预处理函数，用于将文本转换为模型输入格式"""
    text = examples.get('text')
    if not text:  # 过滤空文本
        return None
        
    tokens = tokenizer.tokenize(text)
    max_tokens_len = max_seq_len - 2  # 预留[CLS]和[SEP]
    tokens = tokens[:max_tokens_len]
    
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids += [0] * (max_seq_len - len(input_ids))  # 填充
    token_type_ids = [0] * max_seq_len  # 单句文本，segment id均为0
    
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "label": examples['label']
    }

def create_dataset(dataset, tokenizer, max_seq_len, batch_size, shuffle=True):
    """将数据集转换为tf.data.Dataset格式"""
    # 预处理所有数据
    processed = [preprocess_function(item, tokenizer, max_seq_len) for item in dataset]
    processed = [x for x in processed if x is not None]  # 过滤无效数据
    
    # 转换为tf.data.Dataset
    def generator():
        for item in processed:
            yield (
                {
                    "input_ids": np.array(item["input_ids"], dtype=np.int32),
                    "token_type_ids": np.array(item["token_type_ids"], dtype=np.int32)
                },
                np.array([item["label"]], dtype=np.int32)
            )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "input_ids": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
                "token_type_ids": tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        ),
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(processed))
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    print("正在加载分词器和数据集...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 加载完整数据集
    full_dataset = load_dataset("XiangPan/ChnSentiCorp_htl_8k")['train']
    
    # 划分训练集和测试集（1:5比例）
    print(f"将数据集按{1 - TEST_SIZE:.0%}:{TEST_SIZE:.0%}划分为训练集和测试集...")
    dataset_split = full_dataset.train_test_split(test_size=TEST_SIZE, shuffle=True, seed=42)
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']
    
    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    
    # 创建训练集和测试集的tf.data.Dataset
    train_tf_dataset = create_dataset(
        train_dataset, 
        tokenizer, 
        MAX_SEQ_LEN, 
        BATCH_SIZE, 
        shuffle=True
    )
    test_tf_dataset = create_dataset(
        test_dataset, 
        tokenizer, 
        MAX_SEQ_LEN, 
        BATCH_SIZE, 
        shuffle=False
    )
    
    print("微调数据集已创建。")

    # 3. 加载预训练模型并构建微调模型
    try:
        mini_bert_pretraining = keras.models.load_model('mini_bert_zh_hk.keras', safe_mode=False)
        print("成功加载预训练模型 'mini_bert_zh_hk.keras'")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保 'mini_bert_zh_hk.keras' 文件存在且自定义层 (BertEmbedding, TransformerEncoder) 已正确定义。")
        return
    
    for layer in mini_bert_pretraining.layers:
        layer.trainable = False  # 冻结预训练层

    bert_features = mini_bert_pretraining.get_layer(name="pooled_output").output
    
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(bert_features)

    fine_tuning_model = keras.Model(
        inputs=mini_bert_pretraining.inputs,
        outputs=classification_output,
    )

    fine_tuning_model.summary()

    print("编译模型...")
    fine_tuning_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name='accuracy')]
    )

    print("开始训练微调模型...")
    fine_tuning_model.fit(
        train_tf_dataset,
        epochs=EPOCHS,
        validation_data=test_tf_dataset  # 训练过程中使用测试集作为验证集
    )

    # 在测试集上进行最终评估
    print("\n在测试集上进行最终评估...")
    test_loss, test_acc = fine_tuning_model.evaluate(test_tf_dataset)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    print("微调完成！")
    fine_tuning_model.save('mini_bert_sentiment_classifier.keras')
    print("微调后的模型已保存为 'mini_bert_sentiment_classifier.keras'。")

if __name__ == '__main__':
    main()
