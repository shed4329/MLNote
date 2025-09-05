import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 配置参数（与训练时保持一致）
seq_length = 5  # 序列长度
oov_token = "<UNK>"  # 未知词标记
eop_token = "<EOP>"  # 诗歌结束标记
model_path = 'simple_rnn_tang_poem_generator.h5'  # 模型路径
corpus_file = './corpus.txt'  # 处理后的语料文件


class TangPoemGenerator:
    def __init__(self):
        """初始化唐诗生成器，加载模型和词汇表"""
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型。")

        # 检查语料文件是否存在
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"语料文件 {corpus_file} 不存在，请先运行训练脚本生成。")

        # 加载模型
        print(f"正在加载模型 {model_path}...")
        self.model = load_model(model_path)

        # 加载词汇表并创建映射
        print("正在加载词汇表...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            all_txt = f.read()

        # 创建字符到索引的映射（与训练时保持一致）
        self.vocab = sorted(list(set(all_txt)))
        self.vocab.insert(0, oov_token)
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.oov_idx = self.char_to_idx[oov_token]

        print(f"初始化完成，词汇表大小: {self.vocab_size}")
        print("=" * 50)

    def generate_poem(self, start_text, max_length=200, temperature=1.0):
        """
        生成唐诗

        参数:
            start_text: 起始文本，至少需要seq_length个字符
            max_length: 生成诗歌的最大长度
            temperature: 控制生成的随机性，值越小越确定，越大越随机

        返回:
            生成的唐诗文本
        """
        # 确保起始文本长度足够
        if len(start_text) < seq_length:
            raise ValueError(f"起始文本长度必须至少为 {seq_length} 个字符")

        # 初始化生成文本
        generated = list(start_text)
        current_sequence = start_text[-seq_length:]  # 取最后seq_length个字符作为初始序列

        # 生成文本
        for i in range(max_length):
            # 将当前序列转换为索引
            input_sequence = [self.char_to_idx.get(char, self.oov_idx) for char in current_sequence]
            input_sequence = np.array(input_sequence).reshape(1, -1)  # 重塑为模型输入形状

            # 预测下一个字符
            predictions = self.model.predict(input_sequence, verbose=0)[0][-1]  # 取最后一个时间步的预测

            # 根据温度调整预测分布
            predictions = np.log(predictions) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            # 随机采样下一个字符的索引
            next_index = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char.get(next_index, oov_token)

            # 添加到生成文本
            generated.append(next_char)

            # 如果遇到结束标记，停止生成
            if next_char == eop_token:
                break

            # 更新当前序列
            current_sequence = current_sequence[1:] + next_char

        # 组合生成的文本并返回，移除结束标记
        return ''.join(generated).replace(eop_token, '')

    # 已移除format_poem方法


def main():
    try:
        # 创建生成器实例
        generator = TangPoemGenerator()

        print("欢迎使用唐诗生成器！")
        print(f"请输入至少{seq_length}个字符作为诗歌的开头，程序将为您生成完整的唐诗。")
        print("输入 'exit' 可以退出程序。")

        while True:
            # 获取用户输入
            start_text = input("\n请输入起始文本: ").strip()

            # 检查是否退出
            if start_text.lower() == 'exit':
                print("谢谢使用，再见！")
                break

            # 检查输入长度
            if len(start_text) < seq_length:
                print(f"输入长度不足，请至少输入{seq_length}个字符！")
                continue

            # 获取温度参数
            try:
                temperature = float(input("请输入生成温度(0.1-2.0，值越小越稳定，越大越随机): "))
                if not (0.1 <= temperature <= 2.0):
                    raise ValueError
            except ValueError:
                print("无效的温度值，将使用默认值0.8")
                temperature = 0.8

            # 生成诗歌
            print("\n正在生成诗歌，请稍候...")
            poem = generator.generate_poem(
                start_text,
                max_length=150,
                temperature=temperature
            )

            # 显示结果
            print("\n===== 生成的唐诗 =====")
            print(poem)
            print("======================")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
