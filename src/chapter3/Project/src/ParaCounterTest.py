from src.chapter3.Project.src.ParaCounter import ParaCounter


def main():
    # 示例段落（豆包可能生成的学术文本）
    test_paragraph = """
    The proposed algorithm was designed to optimize resource allocation. 
    Specifically, it employs a greedy approach to minimize computational overhead. 
    However, further research is needed to validate its performance in large-scale systems. 
    For example, the results indicate that the method achieves a 20% improvement in efficiency; 
    nevertheless, scalability remains a critical challenge.
    """

    # 创建计数器实例
    counter = ParaCounter(test_paragraph)

    # 打印段落原文
    print("=== 测试段落 ===")
    print(test_paragraph.strip())
    print("\n=== 分析结果 ===")

    # 1. TTR (词汇多样性)
    print(f"TTR (词汇多样性): {counter.TTR():.4f}")
    print(f"  解释: 比值越高表示词汇越丰富，AI文本通常较低")

    # 2. 连接词分析
    connectives = counter.connectives()
    print(f"\n连接词分析:")
    print(f"  连接词比例: {connectives['connective-ratio']:.4f}")
    print(f"  句首连接词比例: {connectives['connective-starter-ratio']:.4f}")
    print(f"  解释: AI文本可能过度使用特定连接词（如however, specifically）")

    # 3. 句子长度分析
    sentence_length = counter.sentence_length()
    print(f"\n句子长度分析:")
    print(f"  平均长度: {sentence_length['avg_len']:.2f} 词")
    print(f"  标准差: {sentence_length['std_dev']:.2f}")
    print(f"  解释: AI生成的句子长度更均匀（标准差小）")

    # 4. n-gram分析
    # bigrams = counter.n_grams(2)
    # trigrams = counter.n_grams(3)
    # print(f"\n2-gram分析:")
    # print(f"  最常见的5个2-gram:")
    # for ngram, count in bigrams.most_common(5):
    #     print(f"    '{ngram}': {count}次")
    #
    # print(f"\n3-gram分析:")
    # print(f"  最常见的5个3-gram:")
    # for ngram, count in trigrams.most_common(5):
    #     print(f"    '{ngram}': {count}次")
    # print(f"  解释: AI文本可能包含更多重复的n-gram模式")

    # 5. 被动语态分析
    passive_ratio = counter.passive_voice_ratio()
    print(f"\n被动语态比例: {passive_ratio:.2%}")
    print(f"  解释: 学术写作中被动语态常见，但AI可能过度使用（如'was designed to'）")

    # 6. 特殊标点符号分析
    punctuation = counter.special_punctuation()
    print(f"\n特殊标点符号分析:")
    print(f"  标点频率: {punctuation:.6f}")
    print(f"  解释: AI文本可能机械使用标点（如冒号固定用于举例）")


if __name__ == "__main__":
    main()