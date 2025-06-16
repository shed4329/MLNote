import re
from collections import Counter

class ParaCounter:
    # 初始化
    def __init__(self, paragraph):
        self.paragraph = paragraph
        self.original_text = paragraph  # 添加这个属性
        self.words = self._tokenize_words()
        self.sentences = self._tokenize_sentences()

    # 分词
    def _tokenize_words(self):
        """将段落分割为单词列表

        1.移除所有标点符号
        2.转小写
        3.按照空格分割成单词列表
        """
        clean_text = re.sub(r'[^\w\s\']', '', self.paragraph)
        return clean_text.lower().split()

    # 分句
    def _tokenize_sentences(self):
        """将段落分割为句子列表

        1.按照句号，问号，感叹号分割
        2.移除末尾的空白字符
        3.过滤空句子
        """
        sentences = re.split(r'[.!?]\s*', self.paragraph)
        return [s.strip() for s in sentences if s.strip()]

    def TTR(self):
        """
        Type-Token Ratio
        reflect the diversity of word-using
        TTR = unique words/ all words
        """
        if not self.words:
            return 0
        unique_words = len(set(self.words))
        return unique_words / len(self.words)

    def connectives(self):
        """统计连接词的使用情况

        连接词列表包含:
            转折词: however, nevertheless, yet
            递进词: furthermore, moreover, in addition
            因果词: therefore, thus, consequently
            举例词: for example, specifically
            顺序词: firstly, secondly, finally

        返回:
            dict: 包含连接词统计信息的字典
        """
        # 学术写作中常见的连接词列表
        connective_list = [
            "however", "therefore", "moreover", "furthermore",
            "nevertheless", "thus", "consequently", "in addition",
            "for example", "in contrast", "similarly", "specifically",
            "finally", "firstly", "secondly", "lastly", "hence", "accordingly"
        ]

        connective_cnt = 0
        for word in self.words:
            if word in connective_list:
                connective_cnt += 1

        connective_start_cnt = 0
        for sentence in self.sentences:
            if not sentence:
                continue
            first_word = sentence.split()[0].lower()
            if first_word in connective_list:
                connective_start_cnt += 1

        return {
            "connective-ratio": connective_cnt / len(self.words) if self.words else 0,
            "connective-starter-ratio": connective_start_cnt / len(self.sentences) if self.sentences else 0  # 修正分母
        }

    def sentence_length(self):
        """分析句子长度特征

        计算指标:
            1. 平均句子长度
            2. 最短句子长度
            3. 最长句子长度
            4. 句子长度标准差 (反映句子长度的一致性)

        返回:
            dict: 包含句子长度统计信息的字典
        """
        if not self.sentences:
            return {
                "avg_len": 0,
                "min_len": 0,
                "max_len": 0,
                "std_dev": 0
            }

        # 计算每个句子的长度
        sentence_lens = [len(s.split()) for s in self.sentences]

        avg_len = sum(sentence_lens) / len(sentence_lens)
        min_len = min(sentence_lens)
        max_len = max(sentence_lens)

        # 标准差
        variance = sum((length - avg_len) ** 2 for length in sentence_lens) / len(sentence_lens)  # 修正变量名
        std_dev = variance ** 0.5

        return {
            "avg_len": avg_len,
            "min_len": min_len,
            "max_len": max_len,
            "std_dev": std_dev
        }

    # def n_grams(self, n=2):
    #     """统计n-gram频率
    #
    #    参数:
    #        n (int): n-gram的大小，默认为2 (bigram)
    #
    #    返回:
    #        Counter: 包含n-gram及其频率的计数器对象
    #    """
    #
    #     if len(self.words) < n:
    #         return Counter()
    #
    #     # 生成n-gram序列
    #     n_grams = []
    #     for i in range(len(self.words) - n + 1):
    #         n_gram = ' '.join(self.words[i:i+n])  # 用空格连接单词
    #         n_grams.append(n_gram)
    #
    #     # 使用Counter统计
    #     n_gram_counts = Counter(n_grams)
    #     return n_gram_counts

    def passive_voice_ratio(self):
        if not self.sentences:
            return 0

        passive_cnt = 0
        passive_pattern = [
            r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b',
            r'\bis\s+\w+ed\b', r'\bare\s+\w+ed\b',
            r'\bbeen\s+\w+ed\b', r'\bbeing\s+\w+ed\b'
        ]

        for sentence in self.sentences:
            lower_sentence = sentence.lower()  # 修正变量名
            for pattern in passive_pattern:
                if re.search(pattern, lower_sentence):
                    passive_cnt += 1
                    break

        return passive_cnt / len(self.sentences)

    def special_punctuation(self):
        """特殊符号频率"""
        special_punct = [':', '-', '...', ';']  # 修正拼写
        counts = {p: self.original_text.count(p) for p in special_punct}  # 修正方法调用

        total_chars = len(self.original_text)
        total_punct = sum(counts.values())

        frequency = total_punct / total_chars if total_chars > 0 else 0

        return frequency



