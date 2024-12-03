import mujoco
import mujoco.viewer
import torch 
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import threading

# 当前状态  加速度(dim=3)  陀螺仪(dim=3)  四元数(dim=4)
# 目标参数  四元数(dim=4)  速度(dim=3)  角速度(dim=3)  其他参数(dim=5)
# 输出控制指令(关节力矩)(dim=12)
sensor_dim=0
actuator_dim=0
aim_dim=10

input_dim = sensor_dim+aim_dim+5
output_dim = 12

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
        #将数据写入文件（self.time_stamp）,方便以后使用
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        data = {
        "states": self.states,
        "actions": self.actions,
        "rewards": self.rewards,
        "next_states": self.next_states,
        "dones": self.dones,
        }
        
        file_name = f"data_{self.time_stamp}.json"
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

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
        
    #reset
    def reset(self):
        mujoco.mj_resetData(self.m, self.d,0)
        self.DataCollector.clear()
    
    
    def do_random_action(self):
        # 随机动作
        actions = np.random.uniform(-2, 2, 12)
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
        i=0
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                step_start = time.time()
            
                
                mujoco.mj_step(self.m, self.d)

                self._get_state_().to(self.device)
                
                

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)

                viewer.sync()
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step) 
    
    
    #---------------------#
    # #train
    #---------------------#
    
    def Simulate_with_action(self,render=False):
        while True:
            self._get_state_().to(self.device)
            done=False
            i=0
            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                while viewer.is_running() and done==False:
                    
                    step_start = time.time()
                    
                    action , state = self.set_actuator_torque()
                    
                    mujoco.mj_step(self.m, self.d)

                    next_state = self._get_state_().to(self.device)
                    
                    
                    
                    self.Collector.add_transition(state, action, 0, next_state, done)
                    
                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)
                    viewer.sync()
                    time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step) 
                        
                    #解析状态，执行器力矩，得到下一状态
                    i+=1
                    if i>1000:
                        done=True
            
            
            
            
                  
            print("XXXXXXXXXXXXXX")
            self.Collector.clear()   
                    
            # #stop_event.wait()
            
            # #train()
            
            # self.reset()
                    
                    
                    
                    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PolicyNet(nn.Module):
    def __init__(self,n_hiddens, output_dim,device):
        super(PolicyNet, self).__init__()
        self.action_layers =nn.Sequential(
            nn.LazyLinear(n_hiddens,device=device),
            nn.Tanh(),
            nn.Linear(n_hiddens, n_hiddens*2),
            nn.Tanh(),
            nn.Linear(n_hiddens*2, n_hiddens),
            nn.Linear(n_hiddens, output_dim),
        )
        
    def forward(self, x):
        output = self.action_layers(x)
        actions = F.softsign(output)       
        return actions[0]


#创建定时器


class Agent:
    def __init__(self,model_path,sensor_list,actuator_list,Model,device,DataCollector):
        self.model_path=model_path
        self.sensor_list=sensor_list
        self.actuator_list=actuator_list
        self.Model=Model
        self.device=device
        self.DataCollector=DataCollector
        self.robosim=RobotSimulation(model_path,sensor_list,actuator_list,Model,device,DataCollector)
        self.stop_event=threading.Event()
        self.timer=threading.Timer(0.01,self.Simulate_with_action)
        self.timer.start()
        
    def Simulate_with_action(self):
        self.robosim.Simulate_with_action(render=True)
        self.timer=threading.Timer(0.01,self.Simulate_with_action)
        self.timer.start()
        
    def train(self):
        pass
    def stop(self):
        self.stop_event.set()
        self.timer.cancel()  # 停止定时器


data=DataCollector()

model=PolicyNet(64,12,device)
model_path = "/home/wx/WorkSpeac/WorkSpeac/RL/rl/models/google_barkour_v0/scene.xml"
robosim=RobotSimulation(model_path,sensors,actuators,dataCollector=data,Model=model,device=device)
robosim.set_trajectory(torch.tensor([1,0,0,0]),torch.tensor([0,0,0]),torch.tensor([0,0,0]))
robosim.Simulate_with_action(render=True)