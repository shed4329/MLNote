import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def set_random_seeds(seed = 42):
    """固定随机种子"""
    np.random.seed(seed)
    import random
    random.seed(seed)


def clbp_feature(image, R=2, P=8):
    """向量化优化的CLBP特征提取（速度大幅提升）"""
    h, w = image.shape
    padding = R # 边缘填充宽度
    padded = np.pad(image, padding, mode='edge')  # (h+2R, w+2R)，用边缘像素填充

    # 生成邻域点坐标（P个点）
    theta = 2 * np.pi * np.arange(P) / P
    dx = R * np.cos(theta).astype(int)
    dy = R * np.sin(theta).astype(int)

    # 中心分量CLBP-C（向量化）,>128为1，反映明暗
    clbp_c = (image >= 128).astype(np.uint8)  # 直接对原图二值化（无需padding）

    # 提取所有邻域点的像素值（向量化）
    # 生成所有像素的坐标网格 (h, w)
    i, j = np.mgrid[0:h, 0:w]
    # 每个邻域点的坐标 = 中心坐标 + 偏移（加上padding抵消边缘填充）
    neighbors = []
    for k in range(P):
        ni = i + padding + dy[k]
        nj = j + padding + dx[k]
        neighbors.append(padded[ni, nj])  # (h, w)
    neighbors = np.stack(neighbors, axis=-1)  # (h, w, P)：每个像素的P个邻域值

    # 符号分量CLBP-S（向量化），反映相对明暗
    center = image[..., np.newaxis]  # (h, w, 1)：中心像素值
    diffs = neighbors - center  # (h, w, P)：邻域与中心的差值
    s_codes = np.sum((diffs >= 0) * (1 << np.arange(P)), axis=-1).astype(np.uint8)  # (h, w)，符号编码，前边是二进制，后面是权重

    # 幅值分量CLBP-M（向量化），反映差异程度
    m_mask = (diffs != 0)  # 只考虑符号不同的情况
    m_codes = np.sum(m_mask * ((np.abs(diffs) >= 10) * (1 << np.arange(P))), axis=-1).astype(np.uint8)  # (h, w)

    # 计算直方图并归一化
    hist_c = np.histogram(clbp_c, bins=2, range=(0, 1))[0] / (h * w)
    hist_s = np.histogram(s_codes, bins=2 ** P, range=(0, 2 ** P - 1))[0] / (h * w)
    hist_m = np.histogram(m_codes, bins=2 ** P, range=(0, 2 ** P - 1))[0] / (h * w)

    return np.concatenate([hist_c, hist_s, hist_m])

def load_kth_tips(data_root,resize=(128,128)):
    features = []
    labels = []
    classes = []

    if not os.path.exists(data_root):
        raise ValueError(f"数据集路径不存在：{data_root}")
    # 按字母顺序排序类别，确保标签一致性
    for cls in sorted(os.listdir(data_root)):
        cls_dir = os.path.join(data_root, cls)
        if os.path.isdir(cls_dir):
            classes.append(cls)
            label = len(classes) - 1  # 类别标签从0开始

            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue  # 跳过损坏的图像
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img_resized = cv2.resize(img_gray, resize)
                        feat = clbp_feature(img_resized)
                        features.append(feat)
                        labels.append(label)
                    except Exception as e:
                        print(f"处理图像 {img_path} 出错：{e}")
    if len(features) == 0:
        raise ValueError("未加载到任何图像，请检查数据集路径和格式")

    return np.array(features), np.array(labels), classes


# 生成并保存评估报告
def generate_report(y_true, y_pred, classes, save_dir):
    """生成分类报告、混淆矩阵并保存"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 1. 文本报告
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4  # 保留4位小数
    )
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("分类报告已保存")
    print(report)

    # 2. 混淆矩阵可视化
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    print("混淆矩阵已保存")


# 主函数
def main():
    # 配置参数
    DATA_ROOT = "./KTH_TIPS"  # 替换为你的数据集路径
    SAVE_DIR = "./results"  # 报告和模型保存目录
    MODEL_PATH = os.path.join(SAVE_DIR, "clbp_svm_model.pkl")
    SCALER_PATH = os.path.join(SAVE_DIR, "feature_scaler.pkl")
    TEST_SIZE = 0.3  # 测试集比例
    RANDOM_SEED = 42  # 固定种子

    # 固定随机种子
    set_random_seeds(RANDOM_SEED)

    # 加载数据
    print("加载数据集并提取CLBP特征...")
    X, y, classes = load_kth_tips(DATA_ROOT)
    print(f"数据集信息：{X.shape[0]}个样本，{X.shape[1]}维特征，{len(classes)}个类别")
    print(f"类别列表：{classes}")

    # 划分训练集和测试集（ stratify确保类别比例一致 ）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y  # 关键：保持训练集和测试集的类别分布与原始数据一致
    )
    print(f"训练集：{X_train.shape[0]}个样本，测试集：{X_test.shape[0]}个样本")

    # 特征标准化（保存scaler用于后续预测）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM分类器
    print("训练SVM分类器...")
    clf = SVC(
        kernel='rbf',
        C=10,
        gamma=0.1,
        random_state=RANDOM_SEED,  # 分类器固定种子
        probability=True  # 允许后续预测概率
    )
    clf.fit(X_train_scaled, y_train)

    # 预测与评估
    print("评估模型性能...")
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率：{acc:.4f}")

    # 生成并保存评估报告
    generate_report(y_test, y_pred, classes, SAVE_DIR)

    # 保存模型和标准化器
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"模型已保存至：{MODEL_PATH}")
    print(f"特征标准化器已保存至：{SCALER_PATH}")


if __name__ == "__main__":
    main()