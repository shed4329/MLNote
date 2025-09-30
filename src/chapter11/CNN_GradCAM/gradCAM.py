import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 配置参数
MODEL_PATH = 'face_classifier_10k'
IMAGE_PATH = 'sample.jpg'
IMAGE_SIZE = (64, 64)
SAVE_DIR = 'result'
LAST_CONV_LAYER_NAME = 'conv2d_2'

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)


class GradCAM:
    def __init__(self, model_path, last_conv_layer_name):
        # 加载模型
        self.model = tf.keras.models.load_model(model_path)
        self.last_conv_layer_name = last_conv_layer_name

        # 创建梯度计算模型
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

    def preprocess_image(self, img_path):
        """预处理图像"""
        img = image.load_img(img_path)
        img_np = image.img_to_array(img)  # 此时是RGB通道
        img_resized = tf.image.resize(img_np, IMAGE_SIZE)  # 保持RGB
        img_inputs = tf.expand_dims(tf.cast(img_resized, tf.float32) / 255.0, axis=0)
        return img_resized.numpy().astype(np.uint8), img_inputs

    def compute_heatmap(self, img_input):
        """计算Grad-CAM热力图"""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_input)
            target_loss = predictions[:, 0]

        # 计算梯度和通道权重
        grads = tape.gradient(target_loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 加权融合特征图
        conv_outputs = conv_outputs[0]
        cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        cam = tf.nn.relu(cam)

        # 归一化
        if tf.reduce_max(cam) > 0:
            cam = cam / tf.reduce_max(cam)

        return cam.numpy(), predictions[0][0]

    def generate_visualization(self, img_resized, cam):
        """生成可视化结果"""
        # 双插值放大热力图
        heatmap = cv2.resize(cam, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        # 热力图上色：OpenCV生成的是BGR通道
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        # 关键：将BGR转为RGB，确保后续与Matplotlib兼容
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 和原始图像叠加：img_resized是RGB，heatmap_colored已转为RGB，叠加后仍为RGB
        superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
        # superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        return heatmap, superimposed

    def visulaize(self, img_path, save=True):
        """可视化流程"""
        # 预处理：img_resized是RGB通道
        img_resized, img_input = self.preprocess_image(img_path)

        # 计算热力图和预测结果
        cam, pred_prob = self.compute_heatmap(img_input)
        pred_label = 'face' if pred_prob > 0.5 else 'non-face'
        pred_confidence = pred_prob if pred_label == 'face' else 1 - pred_prob

        # 生成可视化结果：superimposed是RGB通道
        heatmap, superimposed = self.generate_visualization(img_resized, cam)

        # 显示结果：确保所有图像都以RGB通道输入Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 原始图像：直接用RGB显示（无需转换）
        axes[0].imshow(img_resized)
        axes[0].set_title(f"original image\nprediction:{pred_label}(fidelity:{pred_confidence:.2f})")
        axes[0].axis('off')

        # 2. 热力图：单通道，用jet色图正常显示
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM heatmap\n(red area:high activation)')
        axes[1].axis('off')

        # 3. 叠加图像：已转为RGB，直接显示（删除原有的RGB2BGR错误转换）
        axes[2].imshow(superimposed)
        axes[2].set_title('heatmap adds original image')
        axes[2].axis('off')

        plt.tight_layout()

        # 保存结果：Matplotlib保存时会自动处理RGB通道，颜色正常
        if save:
            img_name = os.path.basename(img_path).split(".")[0]
            save_path = os.path.join(SAVE_DIR, f"{img_name}_gradcam.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"结果已保存至：{save_path}")

        plt.show()
        return pred_label, pred_confidence


if __name__ == '__main__':
    # 初始化GradCAM
    grad_cam = GradCAM(MODEL_PATH, LAST_CONV_LAYER_NAME)

    # 可视化图像
    print("正在生成Grad-CAM可视化结果...")
    grad_cam.visulaize(IMAGE_PATH, save=True)
    print("可视化完成！")