import json
import os
from src.chapter3.Project.src.ParaCounter import ParaCounter

# 定义输入和输出目录
input_dirs = ["../dataset/raw/human_being", "../dataset/raw/AIGC"]
output_dirs = ["../dataset/processed/human_being", "../dataset/processed/AIGC"]

# 确保输出目录存在，如果不存在则创建
for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

for input_dir, output_dir in zip(input_dirs, output_dirs):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                try:
                    # 构建文件路径
                    input_file_path = os.path.join(root, file)
                    output_file_path = os.path.join(output_dir, file.replace('.txt', '.json'))

                    # 读取文件内容
                    with open(input_file_path, 'r', encoding='utf-8') as input_file:
                        lines = [line.strip() for line in input_file]

                    # 将多行文本合并为一个段落
                    paragraph_text = ' '.join(lines)

                    # 创建ParaCounter实例
                    counter = ParaCounter(paragraph_text)

                    # 收集所有分析结果
                    analysis_results = {
                        "ttr": counter.TTR(),
                        "connectives": counter.connectives(),
                        "sentence_length": counter.sentence_length(),
                        "passive_voice_ratio": counter.passive_voice_ratio(),
                        "special_punctuation": counter.special_punctuation(),
                        # "bigrams": counter.n_grams(),
                        # "trigrams": counter.n_grams(),
                        "AIGC": 1 if "AIGC" in input_dir else 0
                        # 可以添加更多分析指标
                    }

                    # 将Counter对象转换为普通字典（适用于n-gram分析）
                    if hasattr(counter, 'n_grams'):
                        analysis_results["bigrams"] = dict(counter.n_grams(2).most_common(10))
                        analysis_results["trigrams"] = dict(counter.n_grams(3).most_common(10))

                    # 写入JSON文件
                    with open(output_file_path, 'w', encoding='utf-8') as json_file:
                        json.dump(analysis_results, json_file, ensure_ascii=False, indent=4)

                    print(f"分析完成，结果已保存至: {output_file_path}")

                    # 调用分析方法
                    print(f"TTR: {counter.TTR():.4f}")
                    print(f"连接词统计: {counter.connectives()}")
                    # print(f"2-gamma统计:{counter.n_grams()}")
                    # print(f"3-gamma统计:{counter.n_grams(1)}")
                    print(f"被动语态统计:{counter.passive_voice_ratio()}")
                    print(f"特殊符号统计:{counter.special_punctuation()}")
                    print(f"句子长度统计:{counter.sentence_length()}")
                    # 调用其他分析方法...

                except FileNotFoundError:
                    print(f"文件未找到: {input_file_path}")
                except Exception as e:
                    print(f"处理 {input_file_path} 时发生错误: {e}")