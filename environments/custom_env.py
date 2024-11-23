import time
import mujoco
import mujoco.viewer


# 加载 MuJoCo 模型
m = mujoco.MjModel.from_xml_path('/home/wx/WorkSpeac/WorkSpeac/RL/rl/environments/car.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    for i in range(10000):
        step_start = time.time()
        mujoco.mj_step(m, d)

        # 更新可视化选项
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        # 计算下一步所需时间
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:    
            time.sleep(time_until_next_step)
                
