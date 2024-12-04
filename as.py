import numpy as np
import mujoco
import mujoco.viewer
import time
import json

m=mujoco.MjModel.from_xml_path("/home/wx/WorkSpeac/WorkSpeac/RL/rl/environments/models/unitree_go1/scene.xml")
d=mujoco.MjData(m)

def get_state():
    with open('/home/wx/WorkSpeac/WorkSpeac/RL/rl/as.json', 'r') as file:
        data = json.load(file)

    # 提取 "Frames" 数据
    frames = data.get('Frames', [])
    #打印frames[i]的后12个元素
    
    
    return frames


        
        
actions=get_state()   

def clt(d,i):
    acs=actions[i][-12:]
    #倒数第12个元素到倒数第1个元素
    # acs=acs[::-1]
    # print(acs)
    for j in range(12):
        d.ctrl[j]=acs[j]

t=0
i=0
len=500
with mujoco.viewer.launch_passive(m,d) as viewer:
    len=10
    while viewer.is_running() and len>0:
        step_start = time.time()
        
        if t==0:
            len-=1
            t=10
            print(len)
        else:
            t-=1
        
        mujoco.mj_step(m, d)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step) 
            

print("done")