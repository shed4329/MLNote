import random
import gym
from gym.wrappers import RecordVideo
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from collections import namedtuple, deque
import warnings
import gc

# 忽略 deprecation 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 设置TensorFlow内存按需分配（保留此优化）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 录屏目录
if not os.path.exists('dqn_recordings'):
    os.makedirs('dqn_recordings')

# 经验结构
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class BalancedReplayBuffer:
    """平衡型经验回放缓冲区"""

    def __init__(self, capacity=70000):  # 中等容量（介于3万和10万之间）
        self.memory = deque(maxlen=capacity)
        self.state_shape = None
        self.action_size = None

    def set_shapes(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size

    def add(self, state, action, reward, next_state, done):
        # 保持数据类型优化
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.int32(action)
        reward = np.float32(reward)
        done = np.bool_(done)
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        # 预分配数组但保持合理大小
        states = np.empty((batch_size, *self.state_shape), dtype=np.float32)
        actions = np.empty(batch_size, dtype=np.int32)
        rewards = np.empty(batch_size, dtype=np.float32)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.bool_)

        for i, e in enumerate(experiences):
            states[i] = e.state
            actions[i] = e.action
            rewards[i] = e.reward
            next_states[i] = e.next_state
            dones[i] = e.done

        return (states, actions, rewards[:, np.newaxis],
                next_states, dones[:, np.newaxis].astype(np.uint8))

    def __len__(self):
        return len(self.memory)


def build_balanced_q_network(state_size, action_size, layer1_units=48, layer2_units=48):
    """中等规模的Q网络（兼顾性能和速度）"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer1_units, activation='relu', input_shape=(state_size,)),
        tf.keras.layers.Dense(layer2_units, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    return model


class BalancedDQNAgent:
    """平衡型DQN智能体"""

    def __init__(self, state_size, action_size, buffer_size=70000,
                 batch_size=64,  # 恢复到中等批次大小
                 gamma=0.99, learning_rate=1e-3, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every

        # 中等规模网络
        self.Q_network_local = build_balanced_q_network(state_size, action_size)
        self.Q_network_target = build_balanced_q_network(state_size, action_size)

        self.Q_network_local.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

        # 经验缓冲区
        self.memory = BalancedReplayBuffer(capacity=buffer_size)
        self.memory.set_shapes((state_size,), action_size)

        self.t_step = 0

    # 添加tf.function加速推理
    @tf.function
    def _predict_q(self, states):
        return self.Q_network_local(states, training=False)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
                # 减少垃圾回收频率
                if len(self.memory) % 10000 == 0:
                    gc.collect()

    def act(self, state, eps=0.05):
        state = state.reshape(1, -1).astype(np.float32)
        if random.random() > eps:
            # 用tf.function加速预测
            q_values = self._predict_q(state)
            return np.argmax(q_values.numpy()[0])
        else:
            return random.randint(0, self.action_size - 1)

    # 添加tf.function加速学习
    @tf.function
    def _train_step(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q_expected = self.Q_network_local(states, training=True)
            mask = tf.one_hot(actions, self.action_size)
            q_expected = tf.reduce_sum(tf.multiply(q_expected, mask), axis=1, keepdims=True)
            loss = tf.keras.losses.MSE(q_targets, q_expected)

        grads = tape.gradient(loss, self.Q_network_local.trainable_variables)
        self.Q_network_local.optimizer.apply_gradients(
            zip(grads, self.Q_network_local.trainable_variables)
        )
        return loss

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # 批量计算目标Q值（使用完整批次）
        q_targets_next = self.Q_network_target.predict(next_states, verbose=0)
        q_targets_next = np.max(q_targets_next, axis=1).reshape(-1, 1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

        # 使用tf.function加速训练步骤
        self._train_step(states, actions, q_targets)

        # 完整更新目标网络（而非分批）
        self.soft_update(self.Q_network_local, self.Q_network_target, 1e-3)

        # 只清理大对象
        del q_targets_next, q_targets

    def soft_update(self, local_model, target_model, tau=1e-2):
        # 恢复完整更新（速度更快）
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau) * target_weights[i]

        target_model.set_weights(target_weights)
        del local_weights, target_weights


def dqn(env, agent, n_episodes=1200, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995, record_every=20):
    """平衡型训练函数"""
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    original_env = env

    for i_episode in range(1, n_episodes + 1):
        record_this_episode = (i_episode % record_every == 0) or (i_episode == 1)
        current_env = original_env

        if record_this_episode:
            current_env = RecordVideo(
                original_env,
                video_folder='dqn_recordings',
                episode_trigger=lambda x: True,
                name_prefix=f"episode_{i_episode}"
            )

        state = current_env.reset()
        if isinstance(state, tuple):
            state = state[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = current_env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end='')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            # 每100回合回收一次内存
            gc.collect()

        if np.mean(scores_window) >= 200.0:
            print("\n平均分超过200分，任务完成")
            print(f'\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            agent.Q_network_local.save('lunar_lander_dqn_balanced.h5')
            if record_this_episode:
                current_env.close()
            break

        if record_this_episode:
            current_env.close()
            current_env = None

    return scores


def test_agent(agent, num_episodes=5, load_checkpoint=True):
    """测试函数保持不变"""
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
            agent.Q_network_local.load_weights('lunar_lander_dqn_balanced.h5')
            print("模型加载成功，开始测试...")
        except Exception as e:
            print(f"加载模型文件失败: {e}")
            env.close()
            return

    for i in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        score = 0

        while True:
            env.render()
            action = agent.act(state, eps=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

            if done:
                print(f"测试 Episode {i + 1} Score: {score:.2f}")
                break

    env.close()


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    if hasattr(env, 'reset') and 'seed' in env.reset.__code__.co_varnames:
        env.reset(seed=seed)
    else:
        env.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 平衡型智能体参数
    agent = BalancedDQNAgent(
        state_size=state_size,
        action_size=action_size,
        buffer_size=70000,  # 中等缓冲区
        batch_size=64  # 中等批次
    )

    scores = dqn(env=env, agent=agent, record_every=20)

    print("训练分数列表:", scores)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Balanced Training Progress')
    plt.show()

    test_agent(agent=agent)
