import mujoco
import mujoco.viewer
import torch 
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

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
    def __init__(self,xml_path,sensor_list, actuator_list,Model,device):
        
        self.device=device
        
        self.m= mujoco.MjModel.from_xml_path(xml_path)
        self.d=mujoco.MjData(self.m)
        self.Model=Model.to(device)
        
        #目标参数  速度，角速度，四元素
        self.quat = []
        self.vel = []
        self.ang_vel = []
        self.other_params = []

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
            state = self.state.float().to(self.device)  # 将状态移动到CUDA
            actions = self.Model(state)  # 计算动作
            print(actions)
            # 判断actions的长度是否为12, 异常处理
            if actions.size(0) != 12:
                print("执行器力矩维度不正确")
                return
            
            actions = actions.cpu()  # 如果需要将actions移回到CPU
            for actuator_id, torque in zip(self.actuator_ids, actions):
                self.d.ctrl[actuator_id] = torque.item()  # 取出标量值并赋值
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
        self.state = torch.stack([state_a, state_b], dim=0).to(self.device)
        
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
         
    # #---------------------#
    # # #工具函数
    # #-#
    
    def do_random_action(self):
        # 随机动作
        actions = np.random.uniform(-1, 1, 12)
        for actuator_id, torque in zip(self.actuator_ids, actions):
            self.d.ctrl[actuator_id] = torque

    # def pad_to_length(self,tensor, target_length):
    #     """对张量进行填充，使其长度达到 target_length"""
    #     padding = target_length - tensor.size(0)
    #     return torch.cat([tensor, torch.zeros(padding)], dim=0)
    
    
    #---------------------#
    # #仿真
    #---------------------#
    def Simulate(self,render=False):
        self._get_state_()
        if render:
            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                while viewer.is_running():
                    step_start = time.time()

                    self.set_actuator_torque()
                    

                    mujoco.mj_step(self.m, self.d)

                    self._get_state_().to(self.device)

                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)

                    viewer.sync()
                    time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        output = self.action_layers(x)
        actions = torch.tanh(output)       
        return actions[0]

model=PolicyNet(64,12,device)
model_path = "/home/wx/WorkSpeac/WorkSpeac/RL/rl/models/google_barkour_v0/scene.xml"
robosim=RobotSimulation(model_path,sensors,actuators,Model=model,device=device)
robosim.set_trajectory(torch.tensor([1,0,0,0]),torch.tensor([0,0,0]),torch.tensor([0,0,0]))
robosim.Simulate(render=True)
