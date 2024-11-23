import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型的输入维度、输出维度和隐藏层维度
input_dim = None
output_dim = None
hidden_dim = None

#PPO Model

#策略网络
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Policy, self).__init__()
        pass
    
    
    
#价值网络
class Value(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Value, self).__init__()
        pass

#策略网络