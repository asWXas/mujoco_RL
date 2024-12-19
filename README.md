
# 深度强化学习通用框架

## 模块划分

### 1. 环境交互模块
**职责**：
- 管理与强化学习环境的交互，包括状态采集、动作执行、奖励获取。
- 支持单环境或多环境（并行环境）。

**代码**：
```python
class EnvironmentWrapper:
    def __init__(self, env, max_steps=1000):
        self.env = env
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        """重置环境并返回初始状态。"""
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        """执行动作并返回 (下一个状态, 奖励, 是否结束, 其他信息)。"""
        next_state, reward, done, info = self.env.step(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        return next_state, reward, done, info
```

---

### 2. 策略模型模块
**职责**：
- 定义策略网络（Actor）和价值网络（Critic）。

**代码**：
```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """输入状态，返回动作分布和状态价值。"""
        x = self.shared_layers(state)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std.expand_as(mean))
        action_dist = torch.distributions.Normal(mean, std)
        state_value = self.critic(x)
        return action_dist, state_value
```

---

### 3. 采样模块
**职责**：
- 收集与环境交互的轨迹数据。

**代码**：
```python
class Sampler:
    def __init__(self, env_wrapper, model):
        self.env_wrapper = env_wrapper
        self.model = model

    def collect_trajectory(self, steps=2048):
        """采样一批数据并返回轨迹。"""
        pass
```

---

### 4. 优势与回报计算模块
**职责**：
- 使用 GAE 和折扣因子计算优势和目标回报。

**代码**：
```python
class AdvantageCalculator:
    def __init__(self, gamma=0.99, lambda_=0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute(self, rewards, values, dones):
        """计算优势（Advantages）和回报（Returns）。"""
        pass
```

---

### 5. 损失函数模块
**职责**：
- 实现 PPO 或其他算法的优化目标。

**代码**：
```python
class LossFunction:
    def __init__(self, epsilon=0.2, entropy_coeff=0.01):
        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff

    def compute_loss(self, old_log_probs, new_log_probs, advantages, values, returns):
        """计算总损失（策略、价值和熵正则化）。"""
        pass
```

---

### 6. 训练循环模块
**职责**：
- 实现 DRL 的主训练循环。

**代码**：
```python
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
```

---

### 7. 日志与监控模块
**职责**：
- 记录训练过程中的指标，保存和加载模型。

**代码**：
```python
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
```

---

## 通用框架
**职责**：
- 组装模块并启动训练。

**代码**：
```python
def main():
    import gym
    env = gym.make("Pendulum-v1")
    env_wrapper = EnvironmentWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    sampler = Sampler(env_wrapper, model)
    advantage_calculator = AdvantageCalculator()
    loss_fn = LossFunction()

    trainer = Trainer(env_wrapper, model, optimizer, sampler, advantage_calculator, loss_fn)
    logger = Logger()

    trainer.train(num_epochs=100, steps_per_epoch=2048, batch_size=64)
    logger.save_model(model, "./model.pth")


if __name__ == "__main__":
    main()
```
