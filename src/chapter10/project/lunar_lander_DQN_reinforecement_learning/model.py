import random
import gym
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 经验回放存储结构，done表示回合是否结束
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """经验回访缓存区,用于存储和采样经验"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        """添加一条经验到缓冲区"""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """从缓冲区随机采样batch_size条经验"""
        experiences = random.sample(self.memory, k=batch_size)

        # 将经验转换为数组以便输入neural network
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回缓冲区经验数量"""
        return len(self.memory)

def build_q_network(state_size,action_size,layer1_units=64,layer2_units=64):
    """构建Q网络"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer1_units,  activation='relu',input_shape=(state_size,)),
        tf.keras.layers.Dense(layer2_units, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])

    return model

class DQNAgent:
    """ Deep Q Network Agent(DQN Agent) """
    def __init__(self,state_size,action_size,buffer_size=int(1e5),batch_size=64,gamma=0.99,learning_rate=1e-3,update_every=4):
        """初始化智能体"""
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma # 折扣因子
        self.update_every = update_every # 目标网络更新频率

        # Q网络和目标网络
        self.Q_network_local = build_q_network(state_size,action_size)
        self.Q_network_target = build_q_network(state_size,action_size)

        # 编译模型
        self.Q_network_local.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)

        # 计数器，用于定期更新
        self.t_step = 0

    def step(self,state,action,reward,next_state,done):
        """把经验添加到缓冲区，适当的时候更新网络"""
        self.memory.add(state,action,reward,next_state,done)

        # 每update_every步，更新一次网络
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # 如果缓冲区有足够经验，随机采样更新网络
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def act(self,state,eps=0.05):
        """根据当前状态选择动作（epsilon-greedy策略）"""
        state = state.reshape(1,-1)

        # epsilon-greedy策略:有epsilon的概率随机选择动作
        if random.random() >eps:
            q_values = self.Q_network_local.predict(state,verbose=0)
            return np.argmax(q_values[0])
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self,experiences):
        """从经验中学习，更新Q网络"""
        states,actions,rewards,next_states,dones = experiences

        # 获取目标Q值
        q_targets_next = self.Q_network_target.predict(next_states,verbose=0)
        q_targets_next = np.max(q_targets_next,axis=1).reshape(-1,1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

        # 准备训练数据
        with tf.GradientTape() as tape:
            q_expected = self.Q_network_local(states)
            # 使用onehot编码预测对应动作的Q值
            mask = tf.one_hot(actions.squeeze(), self.action_size)
            q_expected = tf.reduce_sum(tf.multiply(q_expected, mask), axis=1, keepdims=True)

            # 计算损失
            loss = tf.keras.losses.MSE(q_targets, q_expected)

        # 反向传播优化
        grads = tape.gradient(loss, self.Q_network_local.trainable_variables)
        self.Q_network_local.optimizer.apply_gradients(
            zip(grads, self.Q_network_local.trainable_variables)
        )

        # 更新目标网络
        self.soft_update(self.Q_network_local,self.Q_network_target,1e-3)

    def soft_update(self,local_model,target_model,tau=1e-2):
        """soft update目标网络参数,θ_target = τ*θ_local + (1-τ)*θ_target"""
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau) * target_weights[i]

        target_model.set_weights(target_weights)

def dqn(env,agent,n_episodes=2000,max_t=1000,eps_start=1.0,eps_end=0.01,eps_decay=0.995):
    """DQN训练"""
    scores = []                         # 每一局的分数
    scores_window = deque(maxlen=100)   # 最近100局的分数,用以计算平均分数
    eps = eps_start                     # epsilon-greedy策略的初始值

    for i_episode in range(1,n_episodes+1):
        state = env.reset()
        # 处理不同gym版本的reset返回值的差异
        if isinstance(state,tuple):
            state = state[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            next_state,reward,done,_ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score+=reward
            if done:
                break
        scores_window.append(score)         # 添加当前得分到滑动窗口
        scores.append(score)                # 添加当前得分到总列表
        eps = max(eps_end,eps_decay*eps)    # 衰减epsilon的值

        # 打印训练进度
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}',end='')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        # 如果连续100局的平均分数超过200分，任务完成
        if np.mean(scores_window) >= 200.0:
            print("平均分超过200分，任务完成")
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # 保存模型
            agent.Q_network_local.save('lunar_lander_dqn.h5')
            break

    return scores

def test_agent(env,agent,num_episodes=5,load_checkpoint=True):
    """测试智能体"""
    if load_checkpoint:
        try:
            agent.Q_network_local.load_weights('lunar_lander_dqn.h5')
        except:
            print("未找到模型文件，无法加载模型")
            return

    for i in range(num_episodes):
        state = env.reset()
        # 处理不同gym返回值差异
        if isinstance(state,tuple):
            state = state[0]
        score = 0
        while True:
            env.render() # 渲染环境
            action = agent.act(state,eps=0.0) # 测试时不使用探索
            next_state,reward,done,_ = env.step(action)
            state = next_state
            score+=reward
            if done:
                print(f"Episode {i + 1} Score: {score:.2f}")
                break
    env.close()

if __name__ == '__main__':
    # 创建环境
    env = gym.make('LunarLander-v2')
    env.seed(seed)

    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 创建智能体
    agnet = DQNAgent(state_size=state_size,action_size=action_size)

    # 训练智能体
    scores = dqn(env=env,agent=agnet)

    print(scores)

    # 绘制训练得分曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # 测试智能体
    test_agent(env=env,agent=agnet)
