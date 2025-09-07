import argparse
import tensorflow as tf
import pickle
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 修正导入，确保包含DecoderLayer
from src.chapter11.translation_Transformer.model import (
    PositionEncoding, EncoderLayer, DecoderLayer,  # 补充DecoderLayer
    CustomSchedule
)


# 定义自定义损失函数（与训练时保持一致）
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none'
    )
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 修复括号缺失
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def load_model_and_tokenizers(model_dir):
    """加载保存的模型和分词器"""
    # 加载模型
    model_path = os.path.join(model_dir, 'final_model')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'PositionEncoding': PositionEncoding,
                'EncoderLayer': EncoderLayer,
                'DecoderLayer': DecoderLayer,  # 修复映射错误
                'CustomSchedule': CustomSchedule,
                'loss_function': loss_function  # 注册损失函数
            }
        )
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")

    # 加载分词器
    en_tokenizer_path = os.path.join(model_dir, 'en_tokenizer.pkl')
    zh_tokenizer_path = os.path.join(model_dir, 'zh_tokenizer.pkl')

    try:
        with open(en_tokenizer_path, 'rb') as f:
            en_tokenizer = pickle.load(f)
        with open(zh_tokenizer_path, 'rb') as f:
            zh_tokenizer = pickle.load(f)
        print("分词器加载成功")
    except Exception as e:
        raise RuntimeError(f"加载分词器失败: {str(e)}")

    return model, en_tokenizer, zh_tokenizer


def interactive_mode(model, en_tokenizer, zh_tokenizer, max_seq_len):
    """交互式翻译模式"""
    print("\n=== 英文到中文翻译工具 ===")
    print("输入英文句子进行翻译，输入 'exit' 退出程序")
    print("-" * 30)

    while True:
        try:
            sentence = input("\n请输入英文: ")
            sentence = sentence.strip()

            if sentence.lower() == 'exit':
                print("感谢使用，再见！")
                break
            if not sentence:
                print("输入不能为空，请重新输入")
                continue

            # 调用model.py中的translate函数进行翻译
            translation = translate(sentence, model, en_tokenizer, zh_tokenizer, max_seq_len)
            print(f"中文翻译: {translation}")
        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"翻译出错: {str(e)}")


def main():
    # 解析命令行参数（仅保留必要参数）
    parser = argparse.ArgumentParser(description='英文到中文交互式翻译工具')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='模型和分词器所在目录 (默认: ./models)')
    parser.add_argument('--max-seq-len', type=int, default=30,
                        help='序列最大长度 (默认: 50)')

    args = parser.parse_args()

    try:
        # 加载模型和分词器
        model, en_tokenizer, zh_tokenizer = load_model_and_tokenizers(args.model_dir)

        # 启动交互式翻译
        interactive_mode(model, en_tokenizer, zh_tokenizer, args.max_seq_len)

    except Exception as e:
        print(f"程序出错: {str(e)}")
        exit(1)


def translate(sentence, transformer_model, en_tokenizer, zh_tokenizer, max_seq_len):
    # 预处理输入句子
    def preprocess_english(text):
        print("\n===== 开始英文预处理 =====")
        original = text
        text = text.strip().lower()
        print(f"去除首尾空格并小写: {text}")

        text = re.sub(r"([?.!,])", r" \1 ", text)
        print(f"标点符号处理后: {text}")

        text = re.sub(r' +', ' ', text)
        print(f"合并空格后: {text}")
        print("===== 英文预处理结束 =====")
        return text

    print("\n===== 翻译流程开始 =====")
    print(f"原始输入句子: {sentence}")

    # 预处理
    processed_sentence = preprocess_english(sentence)
    sentence_list = [processed_sentence]  # 转为列表格式，适应分词器输入
    print(f"预处理后的句子列表: {sentence_list}")

    # 文本转序列
    print("\n===== 文本转序列 =====")
    input_sequence = en_tokenizer.texts_to_sequences(sentence_list)
    print(f"英文分词后的整数序列: {input_sequence}")
    print(f"序列长度: {len(input_sequence[0])}")

    # 序列填充
    input_sequence = pad_sequences(
        input_sequence,
        maxlen=max_seq_len,
        padding='post',
        truncating='post'
    )
    print(f"填充/截断后的序列: {input_sequence}")
    print(f"处理后序列长度: {len(input_sequence[0])}")

    # 初始化目标序列
    print("\n===== 初始化目标序列 =====")
    start_token = zh_tokenizer.word_index["<start>"]
    end_token = zh_tokenizer.word_index["<end>"]
    print(f"<start>标记的ID: {start_token}")
    print(f"<end>标记的ID: {end_token}")

    output_sequence = tf.expand_dims([start_token], 0)
    print(f"初始输出序列: {output_sequence.numpy()}")
    print(f"初始序列对应的文本: {zh_tokenizer.sequences_to_texts(output_sequence.numpy())}")

    # 生成翻译结果
    print("\n===== 开始生成翻译 =====")
    for step in range(max_seq_len):
        print(f"\n----- 生成步骤 {step + 1} -----")
        print('input_sequence=',input_sequence)
        print('output_sequence=',output_sequence)
        # 模型预测
        predictions = transformer_model([input_sequence, output_sequence], training=False)
        print(f"模型输出形状: {predictions.shape}")

        # 提取最后一个时间步的预测
        last_step_predictions = predictions[:, -1:, :]
        print(f"最后一个时间步的预测形状: {last_step_predictions.shape}")

        # 选择概率最高的词
        predicted_id = tf.cast(tf.argmax(last_step_predictions, axis=-1), tf.int32)
        predicted_word = zh_tokenizer.index_word.get(predicted_id.numpy()[0][0], "未知词")
        print(f"预测的词ID: {predicted_id.numpy()}")
        print(f"预测的词语: {predicted_word}")

        # 检查是否到达结束标记
        if predicted_id == end_token:
            print("检测到<end>标记，停止生成")
            break

        # 添加到输出序列
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)
        print(f"当前输出序列: {output_sequence.numpy()}")
        print(f"当前序列对应的文本: {zh_tokenizer.sequences_to_texts(output_sequence.numpy())}")

    # 解码生成的序列
    print("\n===== 翻译结果处理 =====")
    print(f"最终生成的ID序列: {output_sequence.numpy()}")

    translation = zh_tokenizer.sequences_to_texts(output_sequence.numpy())[0]
    print(f"解码后的原始文本: {translation}")

    # 移除特殊标记
    translation = translation.replace("<start>", "").replace("<end>", "").replace("<pad>", "").strip()
    print(f"移除特殊标记后: {translation}")

    # 移除多余空格
    translation = re.sub(r' +', ' ', translation)
    print(f"最终翻译结果: {translation}")
    print("\n===== 翻译流程结束 =====")

    return translation
if __name__ == '__main__':
    main()
