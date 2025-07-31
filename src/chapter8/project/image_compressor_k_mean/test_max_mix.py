import os.path


import numpy as np
from PIL import Image
from tqdm import tqdm

# 配置
lum_weight = 0.4 # 亮度权重
sat_weight = 0.6 # 饱和度权重

def calculate_saturation(rgb):
    """计算RGB饱和度(0-1)"""
    max_channel = np.max(rgb,axis=1)
    min_channel = np.min(rgb,axis=1)
    # 避免除0
    denominator = max_channel+min_channel # 分母
    delta = max_channel-min_channel # 分子

    # 对每个像素单独判断，避免除0（使用np.where实现向量化操作）
    saturation = np.where(
        (denominator < 1e-6) | (delta < 1e-6),  # 对每个像素检查分母是否为0
        0.0,               # 分母为0时饱和度为0
        delta / denominator  # 否则正常计算
    )

    return saturation

def load_image(image_path):
    """加载图片并转换为RGB"""
    img = Image.open(image_path)
    # 转换为RGB格式
    img = img.convert('RGB')
    # 归一化
    img_array = np.array(img,dtype=np.float64)/255.0
    return img_array

def kmeans_compress(image_array,k=16,max_iterations=100):
    """
    使用k-mean压缩图片
    :param image_array: 图片numpy数组
    :param k: 聚类数量
    :param max_iterations: 最大迭代次数
    :return: 压缩后的图片数组和聚类中心
    """

    # 获取图片维度
    height,width,channels = image_array.shape

    # 转换为二维数组
    pixels = image_array.reshape(-1,channels)
    num_pixels = pixels.shape[0]

    # 初始化
    np.random.seed(42)
    indices = np.random.choice(num_pixels,k,replace=False) # 不允许重复抽样
    centroids = pixels[indices].copy()

    # 执行k-mean
    for m in tqdm(range(max_iterations),desc="压缩进度",unit="轮"):
        # 分配聚类中心
        distances = np.sqrt(((pixels-centroids[:,np.newaxis])**2).sum(axis=2))
        # 找到最近的中心
        labels = np.argmin(distances,axis=0)

        # 更新聚类中心
        new_centroids = []
        for i in range(k):
            cluster_pixels = pixels[labels == i] # 第i个聚类中心的像素
            if len(cluster_pixels) > 0:
                # 非空
                new_centroid = cluster_pixels.mean(axis=0).reshape(3,)
            else:
                # 空聚类
                new_centroid = centroids[i].reshape(3,)
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)

        # 是否收敛
        if np.allclose(centroids,new_centroids,atol=1e-6):
            break

        centroids = new_centroids



    # 找出cluster中最亮丽的颜色
    vivid_colors = []
    for i in range(k):
        cluster_pixels = pixels[labels == i] # cluster
        if len(cluster_pixels) == 0:
            # 空聚类
            vivid_color = centroids[i]
        else:
            # Luminosity亮度算法，人眼对RGB的感受程度不一样
            brightness = 0.299 * cluster_pixels[:, 0] + 0.587 * cluster_pixels[:, 1] + 0.114 * cluster_pixels[:, 2]

            # 饱和度优先
            saturation = calculate_saturation(cluster_pixels)

            # 归一化
            norm_saturation = (saturation - saturation.min()) / (saturation.max() - saturation.min() + 1e-9)  # 防止除0
            norm_brightness = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-9)
            # 加权分数
            score = norm_saturation * sat_weight + norm_brightness * lum_weight
            # 找到最亮的像素
            max_score_idx = np.argmax(score)
            # 作为代表色
            vivid_color = cluster_pixels[max_score_idx]
        vivid_colors.append(vivid_color)
    vivid_color_array = np.array(vivid_colors)

    # 用最鲜亮的颜色替换
    compressed_pixels = vivid_color_array[labels]
    # 重塑为原始图形形状
    compressed_array = compressed_pixels.reshape(height,width,channels)

    return compressed_array,centroids

def save_image(image_array,output_path):
    """保存压缩之后的图像"""
    # 回到[0,255]
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(output_path)
    print(f"压缩后的图像已保存至: {output_path}")

def report(original_path,compressed_path):
    """压缩报告"""
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    ratio = compressed_size/original_size
    print(f"原文件大小:{original_size}")
    print(f"压缩后大小:{compressed_size}")
    print(f"压缩比率:{100*ratio:.4f}%")

def compress(input_image,output_image,num_colors):

    # 加载图像
    print("加载图像...")
    original_image = load_image(input_image)

    # 压缩图像
    print(f"使用K-mean进行压缩,保留{num_colors}种颜色...")
    compressed_image, centroids = kmeans_compress(original_image, k=num_colors)
    # 保存图片
    save_image(compressed_image, output_image)
    # 报告
    report(input_image, output_image)

if __name__ == '__main__':
    print("使用max:mix压缩")
    input_image = 'ao.jpg'
    output_images = [
        './compress_mix_max/2.jpg',
        './compress_mix_max/4.jpg',
        './compress_mix_max/8.jpg',
        './compress_mix_max/16.jpg',
        './compress_mix_max/24.jpg',
        './compress_mix_max/32.jpg',
        './compress_mix_max/64.jpg',
    ]

    num_colors = [2,4,8,16,24,32,64]

    for output_image,num_color in zip(output_images,num_colors):
        compress(input_image, output_image, num_color)


