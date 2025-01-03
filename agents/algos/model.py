import torch
import torch.nn as nn
import numpy as np
from agents.Net.actor_critic import *


 
class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):

        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
 
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
 
    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action
 
    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1,1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1,1)
 
        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1-dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value
 
        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []
 
        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
 
        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
 
        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
 
            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
 
            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()



#测试ACT


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from torch.distributions import Normal

# # 定义 Transformer 编码器
# class TransformerEncoder(nn.Module):
#     # Transformer编码器类，继承自nn.Module
#     def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
#         # 初始化函数，定义Transformer编码器的结构
#         super(TransformerEncoder, self).__init__()
#         self.embedding = nn.Linear(input_dim, hidden_dim)  # 线性嵌入层，将输入维度映射到隐藏维度
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,  # 隐藏维度
#             nhead=num_heads,  # 多头注意力机制的头数
#             dim_feedforward=hidden_dim * 4  # 前馈网络的维度
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Transformer编码器

#     def forward(self, x):
#         # 前向传播函数，定义数据通过编码器的流程
#         x = self.embedding(x)  # 将输入数据通过嵌入层
#         x = self.transformer(x)  # 将嵌入后的数据通过Transformer编码器
#         return x  # 返回编码后的数据

# # 定义 ACT 策略网络
# class ACT(nn.Module):
#     # ACT 类，用于实现基于 Transformer 的动作生成模型
#     def __init__(self, state_dim, action_dim, chunk_size, hidden_dim, num_layers, num_heads):
#         # 初始化 ACT 类
#         # state_dim: 状态维度
#         # action_dim: 动作维度
#         # chunk_size: 动作块大小
#         # hidden_dim: 隐藏层维度
#         # num_layers: Transformer 编码器层数
#         # num_heads: Transformer 多头注意力机制的头数
#         super(ACT, self).__init__()
#         # 初始化Transformer编码器，用于处理状态信息
#         self.transformer_encoder = TransformerEncoder(state_dim, hidden_dim, num_layers, num_heads)
#         # 初始化动作解码器，将隐藏层的输出映射到动作空间
#         self.action_decoder = nn.Linear(hidden_dim, action_dim * chunk_size)
#         # 设置每个时间步的动作块大小
#         self.chunk_size = chunk_size
#         # 设置动作空间的维度
#         self.action_dim = action_dim
#     def forward(self, state):
#         # 前向传播函数
#         # state: 输入状态
#         # 返回生成的动作块
#         encoded_state = self.transformer_encoder(state)
#         action_chunk = self.action_decoder(encoded_state[-1])
#         actions = action_chunk.view(-1, self.chunk_size, self.action_dim)
#         return actions

# # 定义值函数网络
# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)

#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         value = self.fc2(x)
#         return value

# # 定义 PPO 算法
# class PPO:
#     def __init__(self, policy, value_net, optimizer, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
#         self.policy = policy
#         self.value_net = value_net
#         self.optimizer = optimizer
#         self.clip_epsilon = clip_epsilon
#         self.value_coef = value_coef
#         self.entropy_coef = entropy_coef

#     def update(self, states, actions, rewards, old_log_probs, advantages):
#         # 计算新策略的动作概率
#         new_actions = self.policy(states)
#         dist = Normal(new_actions, torch.ones_like(new_actions))
#         new_log_probs = dist.log_prob(actions)

#         # 计算 PPO 损失
#         ratio = (new_log_probs - old_log_probs).exp()
#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
#         policy_loss = -torch.min(surr1, surr2).mean()

#         # 计算值函数损失
#         values = self.value_net(states)
#         value_loss = nn.MSELoss()(values, rewards)

#         # 计算熵正则化
#         entropy_loss = -dist.entropy().mean()

#         # 总损失
#         loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

#         # 更新参数
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# def train_ppo_act(env, policy, value_net, optimizer, epochs=100, max_steps=1000, gamma=0.99, lam=0.95):
#     ppo = PPO(policy, value_net, optimizer)
#     for epoch in range(epochs):
#         states, actions, rewards, old_log_probs, values, advantages = [], [], [], [], [], []
#         state = env.reset()
#         for step in range(max_steps):
#             # 生成动作
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             action_chunk = policy(state_tensor)
#             dist = Normal(action_chunk, torch.ones_like(action_chunk))
#             action = dist.sample()
#             log_prob = dist.log_prob(action)

#             # 执行动作
#             next_state, reward, done, _ = env.step(action.squeeze().numpy())

#             # 计算值函数
#             value = value_net(state_tensor)

#             # 存储数据
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             old_log_probs.append(log_prob)
#             values.append(value)

#             state = next_state
#             if done:
#                 break

#         # 计算优势函数
#         rewards = np.array(rewards)
#         values = torch.cat(values).squeeze().detach().numpy()
#         advantages = []
#         gae = 0
#         for t in reversed(range(len(rewards))):
#             delta = rewards[t] + gamma * values[t + 1] - values[t]
#             gae = delta + gamma * lam * gae
#             advantages.insert(0, gae)
#         advantages = torch.FloatTensor(advantages)

#         # 更新策略
#         ppo.update(
#             torch.FloatTensor(np.array(states)),
#             torch.cat(actions),
#             torch.FloatTensor(rewards),
#             torch.cat(old_log_probs),
#             advantages,
#         )
#         print(f"Epoch [{epoch+1}/{epochs}], Steps: {step+1}")

# # Environment
# class DummyEnv:
#     def __init__(self, state_dim, action_dim):
#         self.state_dim = state_dim
#         self.action_dim = action_dim

#     def reset(self):
#         pass

#     def step(self, action):
#         pass

# # 参数设置
# state_dim = 8
# action_dim = 4
# chunk_size = 5
# hidden_dim = 64
# num_layers = 2
# num_heads = 4

# # 初始化模型和环境
# policy = ACT(state_dim, action_dim, chunk_size, hidden_dim, num_layers, num_heads)
# value_net = ValueNetwork(state_dim, hidden_dim)
# optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=1e-3)
# env = DummyEnv(state_dim, action_dim)

# # 训练模型
# train_ppo_act(env, policy, value_net, optimizer, epochs=100)







