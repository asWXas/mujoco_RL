import time
import mujoco
import mujoco.viewer

# 加载 MuJoCo 模型
try:
    model_path = "C:\\Users\\wx\Desktop\\m\\environments\\car.xml"
    m = mujoco.MjModel.from_xml_path(model_path)
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

d = mujoco.MjData(m)

# 启动可视化
with mujoco.viewer.launch_passive(m, d) as viewer:
    for i in range(10000):
        step_start = time.time()
        
        mujoco.mj_step(m, d)

        # 更新可视化选项：切换接触点的显示状态
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        # 计算下一步所需时间
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:    
            time.sleep(time_until_next_step)
