import json
import joblib
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from tkinter.font import Font
import re

from src.chapter3.Project.src.ParaCounter import ParaCounter


def load_model():
    """
    从JSON文件加载逻辑回归模型参数

    返回:
        coefficients: 模型系数(numpy数组)
        intercept: 模型截距(numpy数组)
        feature_names: 特征名称列表
    """
    print("正在加载模型...")
    try:
        with open('./model/model.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        if model_info.get('model_type') != 'LogisticRegression':
            raise ValueError(f"不支持的模型类型: {model_info.get('model_type')}")

        # 提取系数和截距(转换为numpy数组)
        coefficients = np.array(list(model_info['coefficients'].values()))
        intercept = np.array([model_info['intercept']])
        feature_names = list(model_info['coefficients'].keys())

        print("模型加载成功")
        return coefficients, intercept, feature_names

    except FileNotFoundError:
        print(f"错误: 找不到模型文件 './model/model.json'")
        messagebox.showerror("错误", "找不到模型文件 './model/model.json'")
        return None, None, None
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        messagebox.showerror("错误", f"加载模型时发生错误: {e}")
        return None, None, None


def load_scaler():
    print("加载标准化参数...")
    try:
        print("标准化参数加载完毕")
        return joblib.load('./model/scaler.pkl')
    except FileNotFoundError:
        print(f"错误: 找不到标准化文件 './model/scaler.pkl'")
        messagebox.showerror("错误", "找不到标准化文件 './model/scaler.pkl'")
        return None
    except Exception as e:
        print(f"加载标准化文件时发生错误: {e}")
        messagebox.showerror("错误", f"加载标准化文件时发生错误: {e}")
        return None


def sigmoid(x):
    """sigmoid函数，将线性值转换为概率"""
    return 1 / (1 + np.exp(-x))


def extract_features(text):
    """
    从文本中提取特征

    参数:
        text: 输入文本

    返回:
        features: 特征字典
    """
    counter = ParaCounter(text)
    features = {
        "ttr": counter.TTR(),
        "passive_voice_ratio": counter.passive_voice_ratio(),
        "special_punctuation": counter.special_punctuation(),
    }

    connectives = counter.connectives()
    features["connective_ratio"] = connectives["connective-ratio"]
    features["connective_starter_ratio"] = connectives["connective-starter-ratio"]

    sentence_length = counter.sentence_length()
    features["avg_sentence_len"] = sentence_length["avg_len"]
    features["min_sentence_len"] = sentence_length["min_len"]
    features["max_sentence_len"] = sentence_length["max_len"]
    features["sentence_len_std"] = sentence_length["std_dev"]

    return features


def predict(extracted_features, coefficients, intercept, feature_names, scaler):
    """
    预测AIGC概率（包含特征标准化）

    参数:
        extracted_features: 提取的特征字典
        coefficients: 模型系数(numpy数组)
        intercept: 模型截距(numpy数组)
        feature_names: 特征名称列表
        scaler: 训练好的StandardScaler对象

    返回:
        probability: AIGC概率
    """
    if scaler is None:
        return 0.0

    # 1. 将特征字典转换为numpy数组（注意维度）
    feature_vector = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        feature_vector[i] = extracted_features.get(feature, 0)
    feature_vector = feature_vector.reshape(1, -1)  # 转换为(1, n_features)的二维数组

    # 2. 使用scaler进行标准化
    try:
        feature_vector_scaled = scaler.transform(feature_vector)
    except:
        # 如果标准化失败，使用未标准化的数据
        feature_vector_scaled = feature_vector

    # 3. 计算线性组合 z = intercept + sum(coefficient * feature)
    z = intercept[0] + np.dot(feature_vector_scaled, coefficients.T)[0]

    # 4. 通过sigmoid转换为概率
    probability = sigmoid(z)
    return probability


class AIGCTextDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AIGC文本检测工具")
        self.root.geometry("800x800")
        self.root.minsize(600, 500)

        # 设置中文字体
        self.default_font = Font(family="SimHei", size=12)
        self.result_font = Font(family="SimHei", size=10)

        # 加载模型和标准化器
        self.coefficients, self.intercept, self.feature_names = load_model()
        self.scaler = load_scaler()

        # 创建UI
        self._create_ui()

    def _create_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(main_frame, text="AIGC文本检测工具", font=("SimHei", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # 创建文本输入区域
        input_frame = ttk.LabelFrame(main_frame, text="输入文本", padding="5")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, font=self.default_font)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.detect_button = ttk.Button(button_frame, text="AIGC检测", command=self._detect_text)
        self.detect_button.pack(pady=5)

        # 提示信息
        hint_label = ttk.Label(button_frame, text="检测结果仅供参考", font=("SimHei", 9), foreground="gray")
        hint_label.pack(pady=(0, 5))

        # 创建结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="检测结果", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=self.default_font)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)

        # 配置标签样式 - 使用标准十六进制颜色
        self.result_text.tag_configure("normal", foreground="black")
        self.result_text.tag_configure("yellow", foreground="black", background="#FFFF99")  # 浅黄色
        self.result_text.tag_configure("red", foreground="black", background="#FFCCCC")  # 浅红色
        self.result_text.tag_configure("aigc_tag", foreground="blue", font=("SimHei", 9))

    def _detect_text(self):
        """执行文本检测"""
        # 获取输入文本
        text = self.text_input.get("1.0", tk.END).strip()

        if not text:
            messagebox.showwarning("警告", "请输入要检测的文本")
            return

        # 清空结果区域
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        # 按段落分割文本
        paragraphs = re.split(r'\n\s*\n', text)

        # 对每个段落进行检测
        for para in paragraphs:
            if not para.strip():
                self.result_text.insert(tk.END, "\n\n")
                continue

            # 提取特征并预测
            features = extract_features(para)
            prob = predict(features, self.coefficients, self.intercept, self.feature_names, self.scaler)

            # 根据概率选择标签
            if prob <= 0.6:
                tag = "normal"
            elif prob <= 0.8:
                tag = "yellow"
            else:
                tag = "red"

            # 插入段落文本
            self.result_text.insert(tk.END, para, tag)

            # 插入AIGC率标签
            aigc_tag_text = f" [AIGC率:{prob:.2%}]"
            self.result_text.insert(tk.END, aigc_tag_text, "aigc_tag")

            # 插入空行
            self.result_text.insert(tk.END, "\n\n")

        # 禁用结果区域
        self.result_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = AIGCTextDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()