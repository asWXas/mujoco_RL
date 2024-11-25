import mujoco
import mujoco.viewer
import torch 
import numpy as np
import time


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
    "orientation",
    "camera_sensor"
]



class RobotSimulation:
    def __init__(self, m,d,sensor_list, actuator_list,Model):
        
        self.m=m
        self.d=d
        self.Model=Model
        
        #目标参数  速度，角速度，四元素
        self.quat = None
        self.vel = None
        self.ang_vel = None
        self.other_params = None

        #传感器
        self.sensor_names = sensor_list
        self.sensor_start_idxs = None
        self.sensor_ids = None
        self.sensor_dims = None

        #执行器
        self.actuator_names = actuator_list
        self.actuator_ids = None

        #状态
        self.state = None
        
        #初始化
        self._get_sensor_init_()
        self._get_actuator_id_()
        
        
        
    #-------------------#
    # #执行器
    #-------------------#
    
    def _get_actuator_id_(self):
        try:
            # 获取关节所有关节的ID
            self.actuator_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names]
        except Exception as e:
            print(f"获取关节ID时出现错误: {e}")
    
    #-------------------#
    # #设置执行器力矩
    #-#
    
    def set_actuator_torque(self):
        try:
            # 设置执行器的力矩
            actions = self.Model(self.state)[0][0][0]
            #判断actions的长度是否为12,异常处理
            if len(actions)!= 12:
                print("执行器力矩维度不正确")
                return
            for actuator_id, torque in zip(self.actuator_ids, actions):
                self.d.ctrl[actuator_id] = torque
        except Exception as e:
            print(f"设置执行器力矩时出现错误: {e}")

    #-------------------#
    # #设置目标参数
    #-------------------#
    def set_trajectory(self, quat, vel, ang_vel, other_params=None):
        self.quat = quat
        self.vel = vel
        self.ang_vel = ang_vel
        self.other_params = other_params
        
    #-------------------#
    # #传感器
    #-------------------#
    
    def get_sensor_data(self):
        try:
            # 提取传感器数据
            sensor_data_list = [torch.tensor(self.d.sensordata[start_idx:start_idx+dim]) for start_idx, dim in zip(self.sensor_start_idxs, self.sensor_dims)]
            acc = sensor_data_list[0]  # 加速度
            gyro = sensor_data_list[1]  # 陀螺仪
            ori = sensor_data_list[2]  # 方向
        except Exception as e:
            print(f"数据处理出现错误: {e}")
            return None
        return acc, gyro, ori
    
    #-------------------#
    # #状态
    #-#

    def _get_state_(self):
        #状态合并(两个维度：传感器数据和目标参数)
        acc, gyro, ori = self.get_sensor_data()#得到 加速度，角速度，四元素
        state_a = torch.cat([acc, gyro, ori], dim=0)
        state_b = torch.cat([self.vel,self.ang_vel, self.quat], dim=0)
        # 状态合并
        self.state = torch.stack([state_a, state_b], dim=0)
        
        return self.state
    
    
    #-------------------#
    # #初始化
    #-#
    
    def _get_sensor_init_(self):
        start_idxs = []
        dims = []
        try:
            for sensor_name in self.sensor_names:
                sensor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                start_idx = self.m.sensor_adr[sensor_id]
                start_idxs.append(start_idx)
                dim = self.m.sensor_dim[sensor_id]
                dims.append(dim)
        except mujoco.MujocoError as e:
            print(f"MuJoCo错误: {e}")
            return None
        except Exception as e:
            print(f"获取传感器数据时出现错误: {e}")
            return None
        self.sensor_start_idxs = start_idxs
        self.sensor_dims = dims
         
    #---------------------#
    # #工具函数
    #-#

    def pad_to_length(self,tensor, target_length):
        """对张量进行填充，使其长度达到 target_length"""
        padding = target_length - tensor.size(0)
        return torch.cat([tensor, torch.zeros(padding)], dim=0)
    

    #---------------------#
    # #仿真
    #---------------------#
    def Simulate(self,render=False):
        pass



        

model_path = "/home/wx/WorkSpeac/WorkSpeac/RL/rl/models/google_barkour_v0/barkour_v0.xml"
m=mujoco.MjModel.from_xml_path(model_path)
d=mujoco.MjData(m)
robosim=RobotSimulation(m,d,sensors,actuators,Model=None)
robosim.set_trajectory(torch.tensor([1,0,0,0]),torch.tensor([0,0,0]),torch.tensor([0,0,0]))
robosim.Simulate(render=True)








#回报函数
def reward_func(data, senser_data, action):
    # 假设这里定义奖励函数
    pass



# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn


# #继承gym.Env类，实现自己的环境
# class GO1(gym.Env):
#     """创建一个自定义的Gym环境GO1。"""
#     def __init__(self, actions, sensors):
#         """初始化GO1环境，设置动作空间，观察空间和初始状态。"""
#         super(GO1, self).__init__()

#         self.actio_dim = len(actions)
#         self.obv_dim = len(sensors) + self.actio_dim
        
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.actio_dim,), dtype=np.float32)
#         self.observation_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(self.obv_dim,), dtype=np.float32)
        
#         self.observation = None
#         self.reward = None
#         self.done = None
#         self.info = None

#     def step(self, action):
#         """执行给定的动作，更新状态并返回下一个观察值、奖励、是否结束和额外信息。"""
#         return self.observation, self.reward, self.done, self.info

#     def reset(self):
#         """重置环境状态并返回初始观察值。"""
#         return np.zeros(self.obv_dim)  # 示例，实际应为环境重置后的初始状态

#     def render(self, mode='human'):
#         """渲染环境的可视化（当前未实现）。"""
#         pass

# #注册环境
# gym.register(
#     id='GO1-v0',
#     entry_point='GO1:GO1',
    
#     max_episode_steps=1000,
#     reward_threshold=1000.0
# )
# #验证环境
# env = GO1(actions=actuators,sensors=sensors)
# env.reset()
# for i in range(1000):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         env.reset()