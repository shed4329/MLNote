import tensorflow as tf

# 检查tensorflow是否得到CUDA支持
print(tf.test.is_built_with_cuda())

# 检查tensorflow是否可以获取到GPU
print(tf.test.is_gpu_available())