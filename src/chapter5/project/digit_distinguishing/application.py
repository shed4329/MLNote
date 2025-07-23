import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 加载模型
model = tf.keras.models.load_model('./model/model.keras')  # 假设模型保存在当前目录下的model文件夹


def process_image(image_path):
    """处理单个图像并返回预测结果"""
    try:
        # 打开图像
        img = Image.open(image_path)

        # 检查尺寸
        if img.size != (28, 28):
            return False, f"尺寸不匹配: {img.size}, 跳过"

        # 转换为灰度图
        img_gray = img.convert('L')

        # 转换为numpy数组并归一化
        img_array = np.array(img_gray) / 255.0

        # 添加批次维度 (28, 28) -> (1, 28, 28)
        img_batch = np.expand_dims(img_array, axis=0)

        # 模型预测
        logits = model.predict(img_batch)

        # 手动应用softmax将logits转换为概率分布
        probabilities = tf.nn.softmax(logits).numpy()

        predicted_class = np.argmax(probabilities[0])
        confidence = np.max(probabilities[0]) * 100  # 0-100%的合理范围

        return True, f"{os.path.basename(image_path)}\t识别数字: {predicted_class}\t概率: {confidence:.2f}%"

    except Exception as e:
        return False, f"处理失败: {str(e)}"


def main():
    """主函数：处理目录下所有PNG图片"""
    print("手写数字识别程序启动中...")
    print("模型加载完成，开始处理图片...")

    # 获取图片目录
    pic_dir = './pic'

    # 检查目录是否存在
    if not os.path.exists(pic_dir):
        print(f"错误：图片目录 '{pic_dir}' 不存在")
        return

    # 获取所有PNG文件
    png_files = [f for f in os.listdir(pic_dir) if f.lower().endswith('.png')]

    if not png_files:
        print(f"图片目录 '{pic_dir}' 中没有找到PNG文件")
        return

    # 处理每张图片
    for filename in png_files:
        file_path = os.path.join(pic_dir, filename)
        success, result = process_image(file_path)

        if success:
            print(result)
        else:
            print(f"{filename}\t{result}")

    print(f"\n处理完成，共处理 {len(png_files)} 张图片")


if __name__ == "__main__":
    main()