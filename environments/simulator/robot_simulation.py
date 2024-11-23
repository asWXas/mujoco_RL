import mujoco
import mujoco.viewer
import json
import torch 
import numpy as np
import os
import asyncio


# 当前状态  加速度(dim=3)  陀螺仪(dim=3)  四元数(dim=4)
# 目标参数  四元数(dim=4)  速度(dim=3)  角速度(dim=3)  其他参数(dim=5)
# 输出控制指令(关节力矩)(dim=12)
sensor_dim=0
actuator_dim=0
aim_dim=10

input_dim = sensor_dim+aim_dim+5
output_dim = 12



# 定义一个执行器列表，用于表示机器人的不同关节
actuators = [
    "abduction_front_left",
    "hip_front_left",
    "knee_front_left",
    "abduction_hind_left",
    "hip_hind_left",
    "knee_hind_left",
    "abduction_front_right",
    "hip_front_right",
    "knee_front_right",
    "abduction_hind_right",
    "hip_hind_right",
    "knee_hind_right"
]


sensors = [
    "accelerometer",
    "gyro",
    "orientation"
]
#列表长度
print(len(actuators))
print(len(sensors))


#################################################################################################################
# 获取传感器数据
def get_sensor_data(model, sensor_names):    
    # 定义一个空列表，用于存储传感器数据
    start_idxs = []
    dims = []
    try:
        for sensor_name in sensor_names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            start_idx = model.sensor_adr[sensor_id]
            start_idxs.append(start_idx)
            dim = model.sensor_dim[sensor_id]
            dims.append(dim)
    except mujoco.MujocoError as e:
        print(f"MuJoCo错误: {e}")
        return None
    except Exception as e:
        print(f"获取传感器数据时出现错误: {e}")
        return None

    return start_idxs, dims




# 处理数据
def pad_to_length(tensor, target_length):
    """对张量进行填充，使其长度达到 target_length"""
    padding = target_length - tensor.size(0)
    return torch.cat([tensor, torch.zeros(padding)], dim=0)

async def get_data(data, model, start_idxs, dims):
    try:
        # 提取传感器数据
        sensor_data_list = [torch.tensor(data.sensordata[start_idx:start_idx+dim]) for start_idx, dim in zip(start_idxs, dims)]
        acc = sensor_data_list[0]  # 加速度
        gyro = sensor_data_list[1]  # 陀螺仪
        ori = sensor_data_list[2]  # 方向

        # 目标参数
        aim_quat = torch.tensor([0.1, 0.2, 0.3, 0.4])
        aim_vel = torch.tensor([0.1, 0.2, 0.3])
        aim_ang = torch.tensor([0.1, 0.2, 0.3])
        aim_other = torch.zeros(5)

        # 找出所有张量的最大长度
        max_length = max(tensor.size(0) for tensor in [acc, gyro, ori, aim_quat, aim_vel, aim_ang, aim_other])

        # 对齐数据长度
        acc = pad_to_length(acc, max_length)
        gyro = pad_to_length(gyro, max_length)
        ori = pad_to_length(ori, max_length)
        aim_quat = pad_to_length(aim_quat, max_length)
        aim_vel = pad_to_length(aim_vel, max_length)
        aim_ang = pad_to_length(aim_ang, max_length)
        aim_other = pad_to_length(aim_other, max_length)

        # 整合为多通道张量
        # 每个张量作为一个通道
        senser_data = torch.stack([acc, gyro, ori, aim_quat, aim_vel, aim_ang, aim_other], dim=0)  # [channels, max_length]

        # 添加 batch 维度和宽度维度，适配 CNN 输入
        senser_data = senser_data.unsqueeze(0).unsqueeze(-1)  # [1, channels, max_length, 1]

        return senser_data

    except Exception as e:
        print(f"数据处理出现错误: {e}")
        return None

############################################################################################################################




  
    
    
    
    
############################################################################################################################
    
    
# 约束输出
def constrain_output(output):
    # 假设这里对输出进行某种约束操作
    return output  # 具体逻辑待实现

# 预测
def model_predict(senser_data):
    # 假设这里调用某个模型进行推理
    return np.random.rand(12)  # 具体逻辑待实现
    
    
    
# 获取关节所有关节的ID
# 参数 actuators: 包含关节名称的列表
# 返回: 对应关节ID的列表
def get_actuator_id(data, model, actuators):
    try:
        # 使用列表推导式提高效率
        return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuators]
    except Exception as e:
        print(f"获取关节ID时出现错误: {e}")
        return []

# 设置控制输入的函数
def set_control(data, values, actuator_ids):
    try:
        for actuator_id, value in zip(actuator_ids, values):# 直接根据ID设置控制输入，使用zip组合，避免多次索引
            data.ctrl[actuator_id] = value
    except Exception as e:
        print(f"设置控制输入时出现错误: {e}")


# 自定义环境文件，用于控制模型的推理与执行
# 模型推理控制 12个控制维度
async def model_ctrl(data, senser_data, ids):
    try:
        # 读取和解析输出
        output = model_predict(senser_data)

        if output is not None and len(output) >= 12:
            joint_torque = constrain_output(output[:12])
            # 控制指令(关节力矩)
            set_control(data, joint_torque, ids)
        else:
            print("模型预测输出无效或不足12个维度。")
    except Exception as e:
        print(f"控制模型运行出现错误: {e}")


#####################################################################################################################


#回报函数
def reward_func(data, senser_data, action):
    # 假设这里定义奖励函数
    pass



import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


#继承gym.Env类，实现自己的环境
class GO1(gym.Env):
    """创建一个自定义的Gym环境GO1。"""
    def __init__(self, actions, sensors):
        """初始化GO1环境，设置动作空间，观察空间和初始状态。"""
        super(GO1, self).__init__()

        self.actio_dim = len(actions)
        self.obv_dim = len(sensors) + self.actio_dim
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.actio_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(self.obv_dim,), dtype=np.float32)
        
        self.observation = None
        self.reward = None
        self.done = None
        self.info = None

    def step(self, action):
        """执行给定的动作，更新状态并返回下一个观察值、奖励、是否结束和额外信息。"""
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        """重置环境状态并返回初始观察值。"""
        return np.zeros(self.obv_dim)  # 示例，实际应为环境重置后的初始状态

    def render(self, mode='human'):
        """渲染环境的可视化（当前未实现）。"""
        pass

#注册环境
gym.register(
    id='GO1-v0',
    entry_point='GO1:GO1',
    
    max_episode_steps=1000,
    reward_threshold=1000.0
)
#验证环境
env = GO1(actions=actuators,sensors=sensors)
env.reset()
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()