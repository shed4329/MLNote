import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 配置参数
MODEL_PATH = 'face_classifier_10k'
IMAGE_PATH = 'sample.png'
IMAGE_SIZE = (64, 64)
SAVE_DIR = 'result'
LAST_CONV_LAYER_NAME = 'conv2d_2'

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)


class GradCAM:
    def __init__(self, model_path, last_conv_layer_name):
        self.model = tf.keras.models.load_model(model_path)
        self.last_conv_layer_name = last_conv_layer_name
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )
        print('[调试] 模型输出 shape:', self.model.output_shape)

    # ---------- 预处理 ----------
    def preprocess_image(self, img_path):
        img = image.load_img(img_path)
        img_np = image.img_to_array(img)          # RGB
        img_resized = tf.image.resize(img_np, IMAGE_SIZE)
        img_inputs = tf.expand_dims(tf.cast(img_resized, tf.float32) / 255.0, axis=0)
        # ① 保存输入图
        cv2.imwrite(os.path.join(SAVE_DIR, 'debug_input.png'),
                    cv2.cvtColor(img_resized.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        return img_resized.numpy().astype(np.uint8), img_inputs

    # ---------- 热图计算 ----------
    def compute_heatmap(self, img_input):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_input)
            target_loss = predictions[:, 0]

        # ② 卷积输出范围
        print('[调试] conv_outputs  min:', float(tf.reduce_min(conv_outputs)),
              'max:', float(tf.reduce_max(conv_outputs)))

        grads = tape.gradient(target_loss, conv_outputs)
        # ③ 梯度范围
        print('[调试] grads  min:', float(tf.reduce_min(grads)),
              'max:', float(tf.reduce_max(grads)))

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        cam = tf.nn.relu(cam)

        if tf.reduce_max(cam) > 0:
            cam = cam / tf.reduce_max(cam)

        # ④ 原始 cam 范围
        cam_np = cam.numpy()
        print('[调试] cam_raw  min:', cam_np.min(), 'max:', cam_np.max())
        plt.imsave(os.path.join(SAVE_DIR, 'debug_cam_raw.png'), cam_np, cmap='jet')
        return cam_np, predictions[0][0].numpy()

    # ---------- 可视化 ----------
    def generate_visualization(self, img_resized, cam):
        heatmap = cv2.resize(cam, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
        # ⑤ 保存叠加图（debug）
        cv2.imwrite(os.path.join(SAVE_DIR, 'debug_superimposed.png'),
                    cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        return heatmap, superimposed

    # ---------- 主入口 ----------
    def visulaize(self, img_path, save=True):
        img_resized, img_input = self.preprocess_image(img_path)
        cam, pred_prob = self.compute_heatmap(img_input)
        pred_label = 'face' if pred_prob > 0.5 else 'non-face'
        pred_confidence = pred_prob if pred_label == 'face' else 1 - pred_prob
        heatmap, superimposed = self.generate_visualization(img_resized, cam)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title(f"original\n{pred_label} ({pred_confidence:.2f})")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')

        axes[2].imshow(superimposed)
        axes[2].set_title('overlay')
        axes[2].axis('off')

        plt.tight_layout()
        if save:
            out_path = os.path.join(SAVE_DIR, f"{os.path.basename(img_path).split('.')[0]}_gradcam.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print('结果已保存至：', out_path)
        plt.show()
        return pred_label, pred_confidence


if __name__ == '__main__':
    # 初始化GradCAM
    grad_cam = GradCAM(MODEL_PATH, LAST_CONV_LAYER_NAME)

    # 可视化图像
    print("正在生成Grad-CAM可视化结果...")
    grad_cam.visulaize(IMAGE_PATH, save=True)
    print("可视化完成！")