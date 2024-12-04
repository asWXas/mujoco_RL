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


# 当前状态  加速度(dim=3)  陀螺仪(dim=3)  四元数(dim=4)
# 目标参数  四元数(dim=4)  速度(dim=3)  角速度(dim=3)  其他参数(dim=5)
# 输出控制指令(关节力矩)(dim=12)
sensor_dim=0
actuator_dim=0
aim_dim=10

input_dim = sensor_dim+aim_dim+5
output_dim = 12
