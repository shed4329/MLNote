import argparse
import tensorflow as tf
import keras_core as keras
import numpy as np
from transformers import BertTokenizer
# 从fine_tuning.py导入所有自定义组件
from fine_tuning import (
    BertEmbedding, 
    TransformerEncoder,
    ignore_neg100_sparse_categorical_accuracy
)

# 配置参数
MAX_SEQ_LEN = 64
MODEL_PATH = "mini_bert_sentiment_classifier.keras"
POSITIVE_THRESHOLD = 0.7  # 调整正面情感判断阈值为0.7

def load_model_and_tokenizer():
    """加载模型和分词器"""
    try:
        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 加载微调模型，需要指定所有自定义层和函数
        custom_objects = {
            'BertEmbedding': BertEmbedding,
            'TransformerEncoder': TransformerEncoder,
            'ignore_neg100_sparse_categorical_accuracy': ignore_neg100_sparse_categorical_accuracy
        }
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            safe_mode=False
        )
        
        print("模型和分词器加载成功")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型或分词器失败: {str(e)}")
        print(f"请确保模型文件 '{MODEL_PATH}' 存在")
        exit(1)

def preprocess_text(text, tokenizer):
    """预处理文本以适应模型输入"""
    # 分词
    tokens = tokenizer.tokenize(text)
    
    # 截断过长文本
    max_tokens_len = MAX_SEQ_LEN - 2  # 预留 [CLS] 和 [SEP]
    tokens = tokens[:max_tokens_len]
    
    # 添加特殊标记
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # 转换为ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 填充到固定长度
    padding_length = MAX_SEQ_LEN - len(input_ids)
    input_ids += [0] * padding_length  # 0 是 [PAD] 的ID
    
    # 创建token_type_ids（全为0，因为只有一个句子）
    token_type_ids = [0] * MAX_SEQ_LEN
    
    # 转换为numpy数组并添加批次维度
    input_ids = np.array(input_ids, dtype=np.int32)[np.newaxis, :]
    token_type_ids = np.array(token_type_ids, dtype=np.int32)[np.newaxis, :]
    
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids
    }

def predict_sentiment(text, model, tokenizer):
    """预测文本的情感倾向，使用0.7作为正面情感阈值"""
    # 预处理文本
    inputs = preprocess_text(text, tokenizer)
    
    # 预测
    prediction = model.predict(inputs, verbose=0)[0][0]
    
    # 转换为情感标签（使用0.7作为阈值）
    sentiment = "正面" if prediction >= POSITIVE_THRESHOLD else "负面"
    confidence = float(prediction) if sentiment == "正面" else 1 - float(prediction)
    
    return sentiment, confidence

def interactive_mode(model, tokenizer):
    """交互式模式，允许用户输入多个文本进行预测"""
    print(f"\n进入交互式模式（输入 'exit' 退出），正面情感判断阈值: {POSITIVE_THRESHOLD}")
    while True:
        try:
            text = input("\n请输入要分析的中文文本: ")
            if text.lower() == 'exit':
                print("退出程序")
                break
            if not text.strip():
                print("请输入有效的文本")
                continue
                
            sentiment, confidence = predict_sentiment(text, model, tokenizer)
            print(f"情感倾向: {sentiment} (置信度: {confidence:.2%})")
        except KeyboardInterrupt:
            print("\n退出程序")
            break
        except Exception as e:
            print(f"处理时出错: {str(e)}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='中文情感分析工具')
    parser.add_argument('--text', help='要分析的中文文本')
    args = parser.parse_args()
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 如果提供了文本参数，直接分析该文本
    if args.text:
        sentiment, confidence = predict_sentiment(args.text, model, tokenizer)
        print(f"文本: {args.text}")
        print(f"情感倾向: {sentiment} (置信度: {confidence:.2%})，使用的正面阈值: {POSITIVE_THRESHOLD}")
    else:
        # 否则进入交互式模式
        interactive_mode(model, tokenizer)

if __name__ == '__main__':
    main()

