import time
import json
import numpy as np
import torch

class DataCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        #获取当前时间
        self.time_stamp = None
 
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
        # 创建文件名，假设 self.time_stamp 是一个有效的时间戳
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        file_name = f"data_{self.time_stamp}.json"
        
        with open(file_name, 'w') as json_file:
            # 一个一个地写入数据
            for i in range(len(self.states)):
                # 创建当前转移的数据字典
                data = {
                    "states": self.states[i].tolist() if isinstance(self.states[i], (list, torch.Tensor)) else self.states[i],
                    "actions": self.actions[i].tolist() if isinstance(self.actions[i], (list, torch.Tensor)) else self.actions[i],
                    "rewards": self.rewards[i],
                    "next_states": self.next_states[i].tolist() if isinstance(self.next_states[i], (list, torch.Tensor)) else self.next_states[i],
                    "dones": self.dones[i],
                }
                # 写入当前转移的数据，确保是逐个写入
                json.dump(data, json_file, indent=4, ensure_ascii=False)
                json_file.write('\n')  # 每个条目之间换行

        # 清空数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
