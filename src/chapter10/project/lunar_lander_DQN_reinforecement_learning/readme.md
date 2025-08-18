# Project: 月球着陆器

## 项目介绍

基于深度 Q 网络 (DQN) 的月球着陆器 (Lunar Lander) 强化学习解决方案，使用优先经验回放和 Double DQN 技术提升性能。

本项目实现了一个优化的 DQN 智能体，用于解决 OpenAI Gym 中的 LunarLander-v2 环境。智能体通过强化学习算法自主学习如何控制登月舱安全着陆在指定区域。

核心特点：

- 使用优先经验回放 (Prioritized Replay Buffer) 提高样本利用效率
- 采用 Double DQN 技术减少过估计问题
- 自定义奖励函数加速学习过程
- 支持训练过程录屏和结果可视化

## 代码结构
`model.py`: 包含所有核心实现
- `PrioritizedReplayBuffer`: 优先经验回放缓冲区
- `build_64unit_q_network`: Q 网络构建函数
- `OptimizedDQNAgent`: 优化的 DQN 智能体类
- `dqn`: 训练函数
- `test_agent`: 测试函数
- `custom_reward`: 自定义奖励函数

## 使用方法
### 训练智能体
直接运行主程序即可开始训练：
```bash
python model.py
```
训练过程中会：

- 每 100 个回合输出平均得分
- 每 25 个回合录制一次训练视频 (保存至dqn_recordings目录)
- 当 100 个回合的平均得分达到 250 分时自动停止训练
- 保存训练好的模型为lunar_lander_dqn_64units.h5
- 训练结束后显示训练曲线

### 测试智能体
训练完成后，程序会自动进行测试。也可以单独调用测试函数：

运行以下python代码进行测试：
```python
from model import OptimizedDQNAgent, test_agent

state_size = 8  # LunarLander的状态空间维度
action_size = 4  # LunarLander的动作空间维度
agent = OptimizedDQNAgent(state_size, action_size)
test_agent(agent, num_episodes=5)  # 测试5个回合
```
## 算法细节
### 网络结构
两层全连接网络 (128+64 单元)，包含批归一化层加速收敛
### 优先经验回放：
- 根据 TD 误差分配样本优先级
- 重要性采样权重减轻高优先级样本的过度影响
- 逐步增加 beta 值，最终达到 1.0
### 自定义奖励函数：
- 针对不同高度区域设计不同奖励策略
- 鼓励居中、平稳下落和垂直着陆
- 惩罚倾斜过度和速度过快
- 对引擎使用进行适度惩罚，鼓励高效动作
### 训练参数：
- 学习率：5e-4
- 折扣因子 (gamma)：0.99
- 探索率 (epsilon)：从 1.0 衰减至 0.01
- 目标网络更新频率：每 4 步更新一次

测试视频会保存在dqn_recordings/test_episodes目录下。

## 结果
训练完成的智能体理论上能够：

- 稳定地将登月舱降落在着陆台上
- 保持较小的倾斜角度
- 实现软着陆 (垂直速度小)
- 着陆位置接近中心区域

但是我的智能体还没有达到这种程度

## 注意事项
- 训练过程需要 GPU 加速以提高效率
- 由于我是8GB显存，设置了6144MB的显存上限，请根据自己的显卡调整
- 程序会自动管理 GPU 内存，避免内存溢出
- 录屏功能会增加训练时间，可通过调整record_every参数修改录屏频率
- 训练好的模型保存在`lunar_lander_dqn_64units.h5`，可直接用于测试

## 改进方向
- 尝试不同的网络结构 (如添加更多层或单元)
- 调整奖励函数参数以获得更好性能
- 实现 Dueling DQN 或 Rainbow 等更先进的 DQN 变体
- 对比不同参数设置下的训练效率和最终性能

