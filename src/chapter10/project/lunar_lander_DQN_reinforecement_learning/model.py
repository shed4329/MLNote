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
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 禁用TensorFlow数据优化（解决CANCELLED警告）
tf.config.optimizer.set_jit(False)  # 关闭XLA优化
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出

# 随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 录屏目录
if not os.path.exists('dqn_recordings'):
    os.makedirs('dqn_recordings')


class MemoryEfficientReplayBuffer:
    """内存高效的经验回放缓冲区，使用预分配数组存储"""

    def __init__(self, capacity=50000, state_shape=(8,), action_size=4):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_size = action_size

        # 预分配固定大小的数组（内存连续）
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)

        self.size = 0  # 当前存储的经验数量
        self.index = 0  # 下一个要存储的位置

    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区（覆盖旧数据）"""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """采样批次数据（无动态内存分配）"""
        indices = np.random.choice(self.size, batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices].reshape(-1, 1)
        next_states = self.next_states[indices]
        dones = self.dones[indices].reshape(-1, 1).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return self.size


def build_64unit_q_network(state_size, action_size, layer1_units=64, layer2_units=64):
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

    def __init__(self, state_size, action_size, buffer_size=50000,
                 batch_size=64, gamma=0.99, learning_rate=5e-4,  # 降低学习率提高稳定性
                 update_every=2):  # 更频繁更新目标网络
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
        self.memory = MemoryEfficientReplayBuffer(
            capacity=buffer_size,
            state_shape=(state_size,),
            action_size=action_size
        )

        self.t_step = 0

    @tf.function
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
        states, actions, rewards, next_states, dones = experiences

        with tf.device('/GPU:0'):
            # 使用predict_on_batch替代predict，避免数据管道优化
            q_targets_next = self.Q_network_target.predict_on_batch(next_states)
            q_targets_next = np.max(q_targets_next, axis=1).reshape(-1, 1)
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

            # 转换为TensorFlow张量
            states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
            q_targets_tf = tf.convert_to_tensor(q_targets, dtype=tf.float32)

            # 训练步骤
            self._train_step(states_tf, actions_tf, q_targets_tf)

            # 更新目标网络
            self.soft_update(self.Q_network_local, self.Q_network_target, 1e-3)

        # 清理临时变量
        del states, actions, rewards, next_states, dones
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
        if np.mean(scores_window) >= 200.0:
            print("\n平均分超过200分，任务完成")
            print(f'Environment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
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
        buffer_size=50000,
        batch_size=64
    )

    scores = dqn(env=env, agent=agent, record_every=50)

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
