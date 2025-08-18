from datetime import datetime
import random
import gym
from gym.wrappers import RecordVideo
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from collections import deque
import warnings
import gc

# 忽略 deprecation 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 设置TensorFlow内存按需分配并禁用数据优化警告
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # 1. 允许GPU动态增长显存（但取消严格限制）
            tf.config.experimental.set_memory_growth(gpu, True)

            # 2. 可选：设置显存上限（根据你的GPU显存调整，例如8GB显卡设为6144MB）
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],  # 指定使用第0块GPU
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
            )

            # 3. 启用XLA加速（提高GPU计算效率）
            # tf.config.optimizer.set_jit(True)  # 重新开启XLA

    except RuntimeError as e:
        print(e)

# 禁用TensorFlow数据优化（解决CANCELLED警告）
tf.config.optimizer.set_jit(False)  # 关闭XLA优化
# mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出

# 随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 录屏目录
if not os.path.exists('dqn_recordings'):
    os.makedirs('dqn_recordings')


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, state_shape=(8,)):
        self.capacity = capacity
        self.state_shape = state_shape
        self.alpha = 0.6  # 优先级权重（0=随机，1=完全按优先级）
        self.beta = 0.4  # 重要性采样权重（初始值）
        self.beta_increment = 0.001  # 每步增加beta，最终到1.0

        # 预分配数组
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.priorities = np.empty(capacity, dtype=np.float32)  # 优先级（TD误差）

        self.size = 0
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        # 新样本默认最高优先级（避免优先级为0）
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.index] = max_prio

        # 存储样本
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.beta = min(1.0, self.beta + self.beta_increment)  # 逐步增加beta

    def sample(self, batch_size):
        if self.size == 0:
            return None

        # 按优先级采样
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # 计算重要性权重（减少高优先级样本的过度影响）
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化

        # 提取样本
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices].reshape(-1, 1)
        next_states = self.next_states[indices]
        dones = self.dones[indices].reshape(-1, 1).astype(np.uint8)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, errors):
        # 用TD误差更新优先级（误差越大，优先级越高）
        for i, idx in enumerate(indices):
            self.priorities[idx] = abs(errors[i]) + 1e-6  # 加小值避免0优先级

    def __len__(self):
        return self.size


def build_64unit_q_network(state_size, action_size, layer1_units=128, layer2_units=64):
    """64单元网络结构，保持学习能力"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer1_units, activation='relu', input_shape=(state_size,)),
        tf.keras.layers.BatchNormalization(),  # 添加批归一化加速收敛
        tf.keras.layers.Dense(layer2_units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    return model


class OptimizedDQNAgent:
    """优化的DQN智能体，保持64单元网络"""

    def __init__(self, state_size, action_size, buffer_size=100000,
                 batch_size=64, gamma=0.99, learning_rate=5e-4,  # 降低学习率提高稳定性
                 update_every=4):  # 更频繁更新目标网络
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every  # 从4改为2，加速目标网络更新

        # 64单元网络
        self.Q_network_local = build_64unit_q_network(state_size, action_size)
        self.Q_network_target = build_64unit_q_network(state_size, action_size)

        # 使用Adam优化器，调整学习率
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.Q_network_local.compile(
            optimizer=self.optimizer,
            loss='mse'
        )

        # 经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_size,  # 增大容量到10万，增加样本多样性
            state_shape=(state_size,)
        )

        self.t_step = 0

    @tf.function  # XLA编译加速
    def _predict_q(self, states):
        return self.Q_network_local(states, training=False)

    def step(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32).reshape(self.state_size)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(self.state_size)

        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state, eps=0.05):
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if random.random() > eps:
            with tf.device('/GPU:0'):
                q_values = self._predict_q(state)
                action = np.argmax(q_values.numpy()[0])
            return action
        else:
            return random.randint(0, self.action_size - 1)

    @tf.function
    def _train_step(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q_expected = self.Q_network_local(states, training=True)
            mask = tf.one_hot(actions, self.action_size)
            q_expected = tf.reduce_sum(tf.multiply(q_expected, mask), axis=1, keepdims=True)
            loss = tf.keras.losses.MSE(q_targets, q_expected)

        grads = tape.gradient(loss, self.Q_network_local.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.Q_network_local.trainable_variables))
        return loss

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, indices, weights  = experiences

        with tf.device('/GPU:0'):
            # 计算Q_targets（同Double DQN）
            q_local_next = self.Q_network_local.predict_on_batch(next_states)
            best_actions = np.argmax(q_local_next, axis=1)
            q_targets_next = self.Q_network_target.predict_on_batch(next_states)
            q_targets_next = q_targets_next[np.arange(len(q_targets_next)), best_actions].reshape(-1, 1)
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

            # 计算当前Q值和TD误差（用于更新优先级）
            q_expected = self.Q_network_local.predict_on_batch(states)
            q_expected = q_expected[np.arange(len(q_expected)), actions].reshape(-1, 1)
            td_errors = q_expected - q_targets  # TD误差

            # 带权重的损失（重要性采样权重）
            states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
            q_targets_tf = tf.convert_to_tensor(q_targets, dtype=tf.float32)
            weights_tf = tf.convert_to_tensor(weights.reshape(-1, 1), dtype=tf.float32)

            with tf.GradientTape() as tape:
                q_pred = self.Q_network_local(states_tf, training=True)
                mask = tf.one_hot(actions_tf, self.action_size)
                q_pred = tf.reduce_sum(tf.multiply(q_pred, mask), axis=1, keepdims=True)
                loss = tf.reduce_mean(weights_tf * tf.square(q_targets_tf - q_pred))  # 加权损失

            grads = tape.gradient(loss, self.Q_network_local.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.Q_network_local.trainable_variables))

            # 更新样本优先级
            self.memory.update_priorities(indices, td_errors.flatten())

            # 软更新目标网络
            self.soft_update(self.Q_network_local, self.Q_network_target, 1e-3)

        # 清理临时变量
        del states, actions, rewards, next_states, dones, indices, weights
        del q_targets_next, q_targets, states_tf, actions_tf, q_targets_tf
        tf.keras.backend.clear_session()

    def soft_update(self, local_model, target_model, tau=1e-2):
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau) * target_weights[i]

        target_model.set_weights(target_weights)
        del local_weights, target_weights


def dqn(env, agent, n_episodes=1500, max_t=1000,  # 增加最大步数
        eps_start=1.0, eps_end=0.01, eps_decay=0.99,  # 减慢epsilon衰减
        record_every=25):  # 降低录屏频率
    """优化的训练函数"""
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    original_env = env
    video_env = None

    for i_episode in range(1, n_episodes + 1):
        record_this_episode = (i_episode % record_every == 0) or (i_episode == 1)
        current_env = original_env

        if record_this_episode:
            if video_env is not None:
                video_env.close()
                del video_env
                gc.collect()

            video_env = RecordVideo(
                original_env,
                video_folder='dqn_recordings',
                episode_trigger=lambda x: True,
                name_prefix=f"episode_{i_episode}"
            )
            current_env = video_env

        # 初始化回合
        state = current_env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype=np.float32)
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = current_env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            # 应用自定义奖励（核心修改）
            reward = custom_reward(state, action, reward)

            agent.step(state, action, reward, next_state, done)
            state = np.asarray(next_state, dtype=np.float32)
            score += reward

            if done:
                break

        # 记录分数
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # 减慢探索衰减

        # 打印进度（带时间戳格式化）
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end='')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        # 任务完成条件
        if np.mean(scores_window) >= 250.0:
            print("\n平均分超过250分，任务完成")
            print(f'Environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            agent.Q_network_local.save('lunar_lander_dqn_64units.h5')
            if record_this_episode:
                current_env.close()
            break

        # 录屏清理
        if record_this_episode:
            current_env.close()
            del current_env
            gc.collect()

        # 内存清理
        del state, score
        gc.collect()

    # 最终清理
    if video_env is not None:
        video_env.close()
        del video_env
    original_env.close()
    gc.collect()

    return scores

def test_agent(agent, num_episodes=5, load_checkpoint=True):
    """测试函数"""
    env = gym.make('LunarLander-v2')
    if hasattr(env, 'reset') and 'seed' in env.reset.__code__.co_varnames:
        env.reset(seed=seed)
    else:
        env.seed(seed)

    test_dir = os.path.join('dqn_recordings', 'test_episodes')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    env = RecordVideo(
        env,
        video_folder=test_dir,
        episode_trigger=lambda x: True,
        name_prefix="test"
    )

    if load_checkpoint:
        try:
            agent.Q_network_local.load_weights('lunar_lander_dqn_64units.h5')
            print("模型加载成功，开始测试...")
        except Exception as e:
            print(f"加载模型文件失败: {e}")
            env.close()
            return

    for i in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype=np.float32)
        score = 0

        while True:
            env.render()
            action = agent.act(state, eps=0.0)
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            state = np.asarray(next_state, dtype=np.float32)
            score += reward

            if done:
                print(f"测试 Episode {i + 1} Score: {score:.2f}")
                break

        del state, score, next_state, reward, done

    env.close()
    gc.collect()


def custom_reward(state, action, original_reward):
    """
    自定义奖励函数，针对三种极端行为优化
    state解析：[x, y, vx, vy, theta, vtheta, left_leg, right_leg]
    - x: 水平位置（-1.5到1.5）
    - y: 高度（0为地面，最高~1.5）
    - vx: 水平速度
    - vy: 垂直速度（负为下降）
    - theta: 角度（弧度，正负表示左右倾斜）
    - vtheta: 角速度
    - left_leg/right_leg: 是否触地（1=触地）
    """
    # 解析状态变量
    x, y, vx, vy, theta, vtheta, left_leg, right_leg = state

    # 基础奖励（保留原始奖励的核心逻辑）
    reward = original_reward

    if action!=0:
        reward -= 0.02
    if action == 1:  # 左引擎点火（左倾力矩）
        # 1. 惩罚加剧左倾的行为（倾斜越严重，惩罚越重）
        if theta < 0:
            reward -= 0.1 * abs(theta) * 3  # 左倾时点火，惩罚随倾斜度增加（最高约-0.15）
        # 2. 奖励纠正右倾的行为（倾斜越严重，奖励越多）
        elif theta > 0:
            # 微小右倾（0~0.1）：轻微奖励，鼓励微调
            if theta <= 0.1:
                reward += 0.03 * (theta / 0.1)  # 0.03~0.03线性增长
            # 较大右倾（>0.1）：更高奖励，强制纠正
            else:
                reward += 0.05 + 0.02 * (theta - 0.1)  # 0.05~更高（上限0.1）
        # 3. 姿态正时点火（无意义动作）：轻微惩罚
        else:
            reward -= 0.02

    elif action == 3:  # 右引擎点火（右倾力矩）
        # 1. 惩罚加剧右倾的行为
        if theta > 0:
            reward -= 0.1 * abs(theta) * 3  # 右倾时点火，惩罚随倾斜度增加
        # 2. 奖励纠正左倾的行为
        elif theta < 0:
            # 微小左倾（-0.1~0）：轻微奖励
            if theta >= -0.1:
                reward += 0.03 * (abs(theta) / 0.1)  # 0.03~0.03线性增长
            # 较大左倾（<-0.1）：更高奖励
            else:
                reward += 0.05 + 0.02 * (abs(theta) - 0.1)  # 0.05~更高（上限0.1）
        # 3. 姿态正时点火：轻微惩罚
        else:
            reward -= 0.02
    # 1.高空区向中间靠
    if y>0.9:
        reward += (0.5-abs(x))*0.8
        if vy>0:
            reward -= 2*vy
        else:
            reward += 0.1
        reward += 0.2*(1.35-y)
        reward += (0.15-abs(theta))*0.8
        reward += (0.15-abs(vtheta))*0.8
    # 中空区，平稳下落
    elif y>0.4:
        if abs(x) < 0.3:
            reward += (0.3-abs(x))*0.6
        elif abs(x) >0.4:
            reward -= (abs(x)-0.4)
        reward += (0.15-abs(vx))*0.8
        reward += (0.9-y)*0.15
        if vy >0:
            reward -= 4*vy
        elif vy<-0.5:
            reward -= 2 * abs(vy)
        else:
            reward += 0.12
        reward += (0.1-abs(theta))*1.2
        reward += (0.1-abs(vtheta))*1.2
    # 低空区
    else:
        reward += (0.1 - abs(x)) * 1.5
        reward += (0.05 - abs(vx)) * 0.8
        reward += (0.4 - y) * 0.25
        if vy > 0:
            reward -= 8 * vy
        elif vy < -0.5:
            reward -= (4 * abs(vy)-1)
        elif vy < -0.25:
            reward -= 2 * abs(vy)
        else:
            reward += 0.15
        reward += (0.05 - abs(theta)) * 2
        reward += (0.05 - abs(vtheta)) * 2
        # 着陆瞬间（腿触地）的姿态奖励
        if left_leg == 1 or right_leg == 1:
            reward += 1.0
            # 垂直速度小（软着陆）
            if abs(vy) < 0.3:
                reward += 2
            # 水平速度小且位置居中
            if abs(vx) < 0.2:
                reward += 2
            if abs(x) < 0.1:
                reward += 2
            if abs(theta) < 0.05:
                reward += 2
    return reward

if __name__ == '__main__':
    # 创建训练环境
    env = gym.make('LunarLander-v2')
    if hasattr(env, 'reset') and 'seed' in env.reset.__code__.co_varnames:
        env.reset(seed=seed)
    else:
        env.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 64单元网络智能体
    agent = OptimizedDQNAgent(
        state_size=state_size,
        action_size=action_size,
        buffer_size=100000,
        batch_size=128
    )

    print(f"开始时间:{datetime.now()}")
    scores = dqn(env=env, agent=agent, record_every=25)

    # 绘制训练曲线
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('64-Unit Network Training Progress')
    plt.show()

    # 清理
    del env, scores
    gc.collect()

    # 测试
    test_agent(agent=agent)

    # 最终清理
    del agent
    gc.collect()
