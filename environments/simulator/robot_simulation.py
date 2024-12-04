import mujoco
import mujoco.viewer
import torch 
import numpy as np
import time


import sys
from pathlib import Path
sys.path.append(Path(r'./').as_posix())

from agents.Net.actor_critic import *
from agents.Net.dataColl import *
from agents.algos.model import *

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
]


class RobotSimulation:
    def __init__(self,xml_path=None,sensor_list=None, actuator_list=None,dataCollector=None,Model=None,device=None):

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
        self.actions=None

        #状态
        self.state = None
        self.Collector=dataCollector
        
        #频率
        self.fq=10
        
        #初始化
        self._get_sensor_init_()
        self._get_actuator_id_()
        
    def _get_actuator_id_(self):# 获取关节所有关节的ID
        try:
            self.actuator_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names]
        except Exception as e:
            print(f"获取关节ID时出现错误: {e}")
    
    def set_actuator_torque(self):#设置执行器力矩
        try:
            state = self.state.float().to(self.device)  # 将状态转为张量
            self.actions = self.Model(state)  # 计算动作
            # 判断actions的长度是否为12, 异常处理
            if self.actions.size(0) != 12:
                print("执行器力矩维度不正确")
                return
            actions = self.actions.cpu()  # 如果需要将actions移回到CPU
            for actuator_id, torque in zip(self.actuator_ids, actions):
                self.d.ctrl[actuator_id] = torque.item()  # 取出标量值并赋值
            return actions , state
        except Exception as e:
            print(f"设置执行器力矩时出现错误: {e}")

    def set_trajectory(self, quat, vel, ang_vel, other_params=None):#设置轨迹
        self.quat = quat
        self.vel = vel
        self.ang_vel = ang_vel
        self.other_params = other_params
        
    def get_sensor_data(self):#获取传感器数据
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
    
    def _get_state_(self):#获取状态
        #状态合并(两个维度：传感器数据和目标参数)
        acc, gyro, ori = self.get_sensor_data()#得到 加速度，角速度，四元素
        state_a = torch.cat([acc, gyro, ori], dim=0)
        state_b = torch.cat([self.vel,self.ang_vel, self.quat], dim=0)
        # 状态合并
        self.state = torch.stack([state_a, state_b], dim=0).to(self.device)
        return self.state  
    
    def _get_sensor_init_(self):#传感器初始化
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
         
    def reset(self):# 重置
        mujoco.mj_resetData(self.m, self.d,0)
        self.Collector.clear()
    
    def sim_fq(self,fq):# 设置频率
        self.fq=fq
    
    def do_random_action(self):
        # 随机动作
        actions = np.random.uniform(-2, 2, 12)
        for actuator_id, torque in zip(self.actuator_ids, actions):
            self.d.ctrl[actuator_id] = torque


    def get_setp(self):# 奖励函数
        done=False
        reward = 0
        return reward, done

    # def pad_to_length(self,tensor, target_length):
    #     """对张量进行填充，使其长度达到 target_length"""
    #     padding = target_length - tensor.size(0)
    #     return torch.cat([tensor, torch.zeros(padding)], dim=0)

    def Simulate(self,render=False):#仿真
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                step_start = time.time()
            
                mujoco.mj_step(self.m, self.d)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)

                viewer.sync()
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step) 
    
    def Simulate_with_action(self,render=False):#训练
        i=0
        while True:
            self._get_state_().to(self.device)
            done=False
            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                while viewer.is_running() and done==False:
                    
                    step_start = time.time()
                    
                    #频率控制
                    if i==0:
                        action , state = self.set_actuator_torque()
                        mujoco.mj_step(self.m, self.d)
                        next_state = self._get_state_().to(self.device)
                        reward, done = self.get_setp()
                        self.Collector.add_transition(state, action, reward, next_state, done)
                        i=self.fq
                        print("XXXXXXXXXXXXXX")
                    else:
                        mujoco.mj_step(self.m, self.d)
                        i-=1
                        
                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)
                    viewer.sync()
                    time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step) 
                        

            print("XXXXXXXXXXXXXX")
 
                    
            # #stop_event.wait()
            
            # #train()
            self.reset()
                    
                    


data=DataCollector()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=PolicyNet(64,12,device)


model_path = "/home/wx/WorkSpeac/WorkSpeac/RL/rl/environments/models/google_barkour_v0/scene.xml"
robosim=RobotSimulation(model_path,sensors,actuators,dataCollector=data,Model=model,device=device)
robosim.set_trajectory(torch.tensor([1,0,0,0]),torch.tensor([0,0,0]),torch.tensor([0,0,0]))
robosim.Simulate(render=True)