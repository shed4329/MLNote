import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def load_assets(model_path, char_to_idx_path, idx_to_char_path):
    """加载模型和字符映射表"""
    try:
        # 加载模型
        model = load_model(model_path)
        print(f"成功加载模型: {model_path}")

        # 加载字符映射
        with open(char_to_idx_path, 'rb') as f:
            char_to_idx = pickle.load(f)

        with open(idx_to_char_path, 'rb') as f:
            idx_to_char = pickle.load(f)

        print("成功加载字符映射表")
        return model, char_to_idx, idx_to_char

    except Exception as e:
        print(f"加载资源失败: {str(e)}")
        return None, None, None


def generate_text(model, char_to_idx, idx_to_char, seed_text, length=100, temperature=1.0):
    """
    根据种子文本生成新文本
    """
    n_vocab = len(char_to_idx)
    generated = seed_text

    # 确保种子文本长度符合模型要求
    seq_length = model.input_shape[1]
    if len(seed_text) < seq_length:
        print(f"警告: 种子文本长度小于模型要求的{seq_length}个字符，将用空格填充")
        seed_text = seed_text.ljust(seq_length)
    else:
        seed_text = seed_text[-seq_length:]  # 取最后seq_length个字符

    current_sequence = seed_text

    for i in range(length):
        # 准备输入序列
        x = np.zeros((1, seq_length, 1))
        for t, char in enumerate(current_sequence):
            if char in char_to_idx:
                x[0, t, 0] = char_to_idx[char] / n_vocab
            else:
                # 处理未见过的字符，用0代替
                x[0, t, 0] = 0

        # 预测下一个字符
        predictions = model.predict(x, verbose=0)[0]

        # 应用温度调整概率分布
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # 根据概率选择下一个字符
        index = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char.get(index, '')

        # 更新生成的文本和当前序列
        generated += next_char
        current_sequence = current_sequence[1:] + next_char

        # 显示进度
        if i % 50 == 0 and i > 0:
            print(f"已生成 {i}/{length} 个字符", end="\r")

    print("\n生成完成")
    return generated


def get_valid_path(prompt, default, file_extension=None):
    """获取有效的文件路径"""
    while True:
        path = input(f"{prompt} (默认: {default}): ").strip()
        if not path:
            path = default

        if os.path.exists(path):
            if file_extension and not path.endswith(file_extension):
                print(f"文件必须是{file_extension}格式")
                continue
            return path
        else:
            print(f"错误: 文件 '{path}' 不存在")
            retry = input("是否使用默认路径? (y/n): ").strip().lower()
            if retry == 'y':
                return default


def get_positive_integer(prompt, default):
    """获取有效的正整数"""
    while True:
        try:
            value = input(f"{prompt} (默认: {default}): ").strip()
            if not value:
                return default
            value = int(value)
            if value > 0:
                return value
            else:
                print("请输入一个正整数")
        except ValueError:
            print("请输入一个有效的整数")


def get_temperature(prompt, default):
    """获取有效的温度值（0.1-5.0之间）"""
    while True:
        try:
            value = input(f"{prompt} (默认: {default}，范围: 0.1-5.0): ").strip()
            if not value:
                return default
            value = float(value)
            if 0.1 <= value <= 5.0:
                return value
            else:
                print("请输入0.1到5.0之间的数值")
        except ValueError:
            print("请输入一个有效的数字")


def main():
    print("=" * 50)
    print("        LSTM 文本生成器        ")
    print("=" * 50)
    print("这个程序将使用已训练的模型生成文本")
    print("请按照提示输入相关信息\n")

    # 获取模型和映射文件路径
    model_path = get_valid_path("请输入模型文件路径", "model.h5", ".h5")
    char_to_idx_path = get_valid_path("请输入char_to_idx映射文件路径", "char_to_idx.pkl", ".pkl")
    idx_to_char_path = get_valid_path("请输入idx_to_char映射文件路径", "idx_to_char.pkl", ".pkl")

    # 加载模型和映射表
    model, char_to_idx, idx_to_char = load_assets(model_path, char_to_idx_path, idx_to_char_path)

    if not model or not char_to_idx or not idx_to_char:
        print("无法继续，程序退出")
        return

    # 获取生成参数
    print("\n--- 生成参数设置 ---")
    seed_text = input("请输入起始文本（种子）: ").strip()
    while not seed_text:
        print("起始文本不能为空")
        seed_text = input("请输入起始文本（种子）: ").strip()

    length = get_positive_integer("请输入生成文本的长度", 200)
    temperature = get_temperature("请输入温度参数（值越小越确定，越大越随机）", 1.0)

    # 生成文本
    print("\n开始生成文本...")
    generated_text = generate_text(
        model,
        char_to_idx,
        idx_to_char,
        seed_text,
        length,
        temperature
    )

    # 显示结果
    print("\n" + "=" * 50)
    print("生成的文本:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

    # 询问是否保存结果
    save = input("\n是否将生成的文本保存到文件? (y/n): ").strip().lower()
    if save == 'y':
        while True:
            output_path = input("请输入保存路径（例如: output.txt）: ").strip()
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(generated_text)
                    print(f"文本已成功保存到: {output_path}")
                    break
                except Exception as e:
                    print(f"保存失败: {str(e)}")
                    retry = input("是否重试? (y/n): ").strip().lower()
                    if retry != 'y':
                        break

    print("\n程序执行完毕，谢谢使用！")


if __name__ == '__main__':
    main()
