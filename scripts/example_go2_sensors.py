"""Go2のMJCFに書かれた<sensor>(gyro/accelerometer/orientation...)と足裏の接触力を、Isaac Lab形式で読む最小例。事前に `cmake --build build --target _mjmlx_rl_native` が必要。

    .venv/bin/python scripts/example_go2_sensors.py --envs 64
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mjmlx_go2_env import FOOT_BODY_IDX  # noqa: E402
from mjmlx_sensors import ContactSensor, Imu, SensorHub  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--build-dir", default="build")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
sys.path.insert(0, args.build_dir)
import _mjmlx_rl_native as native  # noqa: E402

from mjmlx_go2_env import MjmlxGo2Env  # noqa: E402

# 学習envと同じMJCF(go2_mjx.xml由来、<sensor>入り)・同じ初期化でsimを作り、その上にセンサを載せる
env = MjmlxGo2Env(native, args.envs, 0.2, 1.0, use_gpu=not args.cpu, flat_terrain=True, seed=0)
sim, model = env._sim, env._model
hub = SensorHub(sim, model)
imu = Imu(hub)  # go2_mjx.xmlの<sensor>名が既定
feet = ContactSensor(sim, args.envs, FOOT_BODY_IDX, force_threshold=1.0)

zero_action = torch.zeros(args.envs, 12)
for _ in range(100):
    env.step(zero_action)  # 内部で4物理step(dt=0.005)
    hub.read()
    feet.update(0.02)

print("sensors:", hub.names)
print("ang_vel_b      ", imu.ang_vel_b[0].numpy().round(3))
print("projected_grav ", imu.projected_gravity_b[0].numpy().round(3), "(直立なら[0,0,-1]付近)")
print("lin_acc_b      ", imu.lin_acc_b[0].numpy().round(3), "(静止時は+9.81 z付近)")
print("feet net force ", feet.net_forces_w[0].norm(dim=-1).numpy().round(1), "air_time", feet.current_air_time[0].numpy().round(3))
