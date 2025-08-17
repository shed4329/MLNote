import random
import gym
from gym.wrappers import RecordVideo
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from collections import namedtuple, deque
import warnings

# 忽略 deprecation 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 设置随机种子，确保实验可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 创建录屏保存目录
if not os.path.exists('dqn_recordings'):
    os.makedirs('dqn_recordings')

# 经验回放回放存储结构
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """经验回放回放缓冲区"""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([e.action for e in experiences])
        rewards = np.vstack([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def build_q_network(state_size, action_size, layer1_units=64, layer2_units=64):
    """构建Q网络"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer1_units, activation='relu', input_shape=(state_size,)),
        tf.keras.layers.Dense(layer2_units, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    return model


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_size, action_size, buffer_size=int(1e5), batch_size=64,
                 gamma=0.99, learning_rate=1e-3, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every

        # Q网络
        self.Q_network_local = build_q_network(state_size, action_size)
        self.Q_network_target = build_q_network(state_size, action_size)

        self.Q_network_local.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state, eps=0.05):
        state = state.reshape(1, -1)
        if random.random() > eps:
            q_values = self.Q_network_local.predict(state, verbose=0)
            return np.argmax(q_values[0])
        else:
            # 修复随机数非整数警告
            return random.randint(0, self.action_size - 1)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.Q_network_target.predict(next_states, verbose=0)
        q_targets_next = np.max(q_targets_next, axis=1).reshape(-1, 1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

        with tf.GradientTape() as tape:
            q_expected = self.Q_network_local(states)
            mask = tf.one_hot(actions.squeeze(), self.action_size)
            q_expected = tf.reduce_sum(tf.multiply(q_expected, mask), axis=1, keepdims=True)
            loss = tf.keras.losses.MSE(q_targets, q_expected)

        grads = tape.gradient(loss, self.Q_network_local.trainable_variables)
        self.Q_network_local.optimizer.apply_gradients(
            zip(grads, self.Q_network_local.trainable_variables)
        )

        self.soft_update(self.Q_network_local, self.Q_network_target, 1e-3)

    def soft_update(self, local_model, target_model, tau=1e-2):
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau) * target_weights[i]

        target_model.set_weights(target_weights)


def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, record_every=10):
    """训练主函数 - 兼容旧版Gym"""
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    original_env = env

    for i_episode in range(1, n_episodes + 1):
        record_this_episode = (i_episode % record_every == 0) or (i_episode == 1)
        current_env = original_env
        video_recorder = None

        # 仅在需要录制时创建录屏环境
        if record_this_episode:
            # 移除disable_logger参数，兼容旧版本
            current_env = RecordVideo(
                original_env,
                video_folder='dqn_recordings',
                episode_trigger=lambda x: True,
                name_prefix=f"episode_{i_episode}"
            )

        # 初始化回合（使用新的seed设置方式）
        state = current_env.reset(seed=seed) if hasattr(current_env,
                                                        'reset') and 'seed' in current_env.reset.__code__.co_varnames else current_env.reset()
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

        if np.mean(scores_window) >= 200.0:
            print("\n平均分超过200分，任务完成")
            print(f'\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            agent.Q_network_local.save('lunar_lander_dqn.h5')
            if record_this_episode:
                current_env.close()
            break

        # 显式关闭录屏环境，防止递归错误
        if record_this_episode:
            current_env.close()
            # 手动解除引用，防止__del__导致的递归错误
            current_env = None

    return scores


def test_agent(agent, num_episodes=5, load_checkpoint=True):
    """测试函数 - 显示窗口"""
    env = gym.make('LunarLander-v2')
    # 使用新的seed设置方式
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
            agent.Q_network_local.load_weights('lunar_lander_dqn.h5')
            print("模型加载成功，开始测试...")
        except Exception as e:
            print(f"加载模型文件失败: {e}")
            env.close()
            return

    for i in range(num_episodes):
        state = env.reset()
        score = 0

        while True:
            env.render()  # 测试时显示窗口
            action = agent.act(state, eps=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

            if done:
                print(f"测试 Episode {i + 1} Score: {score:.2f}")
                break

    env.close()


if __name__ == '__main__':
    # 创建训练环境
    env = gym.make('LunarLander-v2')
    # 适配新版本seed设置
    if hasattr(env, 'reset') and 'seed' in env.reset.__code__.co_varnames:
        env.reset(seed=seed)
    else:
        env.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size)
    scores = dqn(env=env, agent=agent, record_every=10)

    print("训练分数列表:", scores)

    # 绘制训练曲线
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Progress')
    plt.show()

    # 测试时显示窗口
    test_agent(agent=agent)
