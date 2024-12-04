import sys
from pathlib import Path
sys.path.append(Path(r'./').as_posix())

from agents.Net.actor_critic import *
from agents.Net.dataColl import *
from agents.algos.model import *
from environments.simulator.robot_simulation import *


data=DataCollector()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=PolicyNet(64,12,device)


model_path = "/home/wx/WorkSpeac/WorkSpeac/RL/rl/environments/models/google_barkour_v0/scene.xml"
robosim=RobotSimulation(model_path,sensors,actuators,dataCollector=data,Model=model,device=device)
robosim.set_trajectory(torch.tensor([1,0,0,0]),torch.tensor([0,0,0]),torch.tensor([0,0,0]))
robosim.Simulate_train(render=True)