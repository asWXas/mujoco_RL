import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 定义模型的输入维度、输出维度和隐藏层维度
input_dim = None
output_dim = None
hidden_dim = None

states_type={
    "senser_state",
    "aim_state"
}


# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #
 
class PolicyNet(nn.Module):
    def __init__(self,n_hiddens, output_dim,device):
        super(PolicyNet, self).__init__()
        self.action_layers =nn.Sequential(
            nn.LazyLinear(n_hiddens,device=device),
            nn.ELU(),
            nn.Linear(n_hiddens, n_hiddens*2),
            nn.ReLU(),
            nn.Linear(n_hiddens*2, n_hiddens),
            nn.Linear(n_hiddens, output_dim),
        )
        
    def forward(self, x):
        x=x.float()
        output = self.action_layers(x)
        actions = torch.tanh(output)             
        return actions[0][0][0]
    
# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #
 
class ValueNet(nn.Module):
    def __init__(self, n_hiddens,device):
        super(ValueNet, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(n_hiddens,device=device),
            nn.ELU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.Linear(n_hiddens, 1)
        )
        
    def forward(self, x):
        x=x.float()
        output = self.linear_layers(x)
        return output
    


#数据收集器
class DataCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
 
    def add_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
 
    def get_transition_dict(self):
        transition_dict = {
            'states': torch.tensor(np.array(self.states), dtype=torch.float32),  # 转换为张量
            'aim_states': torch.tensor(np.array(self.next_states), dtype=torch.float32),  # 转换为张量
            'actions': torch.tensor(np.array(self.actions), dtype=torch.float32),  # 转换为张量
            'rewards': torch.tensor(np.array(self.rewards), dtype=torch.float32),  # 转换为张量
            'next_states': torch.tensor(np.array(self.next_states), dtype=torch.float32),  # 转换为张量
            'dones': torch.tensor(np.array(self.dones), dtype=torch.float32)  # 转换为张量
        }
        return transition_dict
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

Collector = DataCollector()
#假设已经收集好数据
data_list = Collector.get_transition_dict()
for data in data_list:
    states=data['states']
    aim_states=data['aim_states']
    actions=data['actions']
    rewards=data['rewards']
    next_states=data['next_states']
    dones=data['dones']
    
    #计算
    #反响传播
    #优化器更新参数
    
    
#----------------------------------- #
# 4. 优势与回报计算模块
#       职责：
#        使用 GAE 和折扣因子计算优势和目标回报。
# --------------------------------- -- #  
class AdvantageCalculator:
    def __init__(self, gamma=0.99, lambda_=0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute(self, rewards, values, dones):
        """计算优势（Advantages）和回报（Returns）。"""
        pass 


#----------------------------------- #
# 5. 损失函数模块
#       职责：
#        计算策略网络、价值网络和熵正则化的损失函数。
# --- -- #  

class LossFunction:
    def __init__(self, epsilon=0.2, entropy_coeff=0.01):
        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff

    def compute_loss(self, old_log_probs, new_log_probs, advantages, values, returns):
        """计算总损失（策略、价值和熵正则化）。"""
        pass
    
    
#----------------------------------------------#
# 6. 训练模块
#       职责：
#        训练策略网络、价值网络和熵正则化。
# - #  
class Logger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir

    def log(self, step, metrics):
        """记录训练指标（如奖励、损失）。"""
        pass

    def save_model(self, model, path):
        """保存模型参数。"""
        pass

    def load_model(self, model, path):
        """加载模型参数。"""
        pass
    
    
#-------------------------------------------------------------#
# 7. 训练过程
#       职责：
#        整合前面的模块，实现训练过程。
# - #  
class Trainer:
    def __init__(self, env, model, optimizer, sampler, advantage_calculator, loss_fn):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.sampler = sampler
        self.advantage_calculator = advantage_calculator
        self.loss_fn = loss_fn

    def train(self, num_epochs=10, steps_per_epoch=2048, batch_size=64):
        """主训练循环。"""
        pass
    
#------------------------------------------------------------#

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
    def predict(self, state):
        action_list = self.actor(state)
        #维度转换为一维
        action_list = action_list.cpu().detach().numpy().flatten()
        
        return action_list
 
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