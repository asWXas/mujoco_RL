import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型的输入维度、输出维度和隐藏层维度
input_dim = None
output_dim = None
hidden_dim = None

# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #
 
class PolicyNet(nn.Module):
    def __init__(self, input_dim, n_hiddens, output_dim):
        super(PolicyNet, self).__init__()
        self.action_layers =nn.Sequential(
            nn.Linear(input_dim, n_hiddens),
            nn.ELU(),
            nn.Linear(n_hiddens, n_hiddens*2),
            nn.ReLU(),
            nn.Linear(n_hiddens*2, n_hiddens),
            nn.Linear(n_hiddens, output_dim),
        )
        
    def forward(self, x):
        output = self.action_layers(x)
        actions = torch.tanh(output)        
        return actions
        
        
        
        
 
# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #
 
class ValueNet(nn.Module):
    def __init__(self, input_dim, n_hiddens):
        super(ValueNet, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, 1)
        )
        
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x
 
# ----------------------------------- #
# 构建模型
# ----------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(ActorCritic, self).__init__()
        self.policy_net = PolicyNet(n_states, n_hiddens, n_actions)
        self.value_net = ValueNet(n_states, n_hiddens)
    def forward(self, x):
        policy = self.policy_net(x)
        value = self.value_net(x)
        return policy, value
    
    
import torch
import torch.nn as nn

# 定义一个Tanh激活函数层
tanh_layer = nn.Tanh()

# 示例输入
input_tensor = torch.tensor([-1.5, 0.0, ])
output_tensor = tanh_layer(input_tensor)

print(output_tensor)  # 输出结果
