#!/usr/bin/env python3
"""Vulkan->CUDA interopのRGB/depthが、CPU読み出し(cam.rgb()/depth())とビット一致することを確認する。非対応環境(torch+cuda無し、interop無し)は77でskip。

usage: test_camera_cuda.py BUILD_DIR
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
os.environ["MJMLX_GO2_XML_SUBST"] = (
    '<camera name="track" pos="0.846 -1.465 0.916" xyaxes="0.866 0.500 0.000 -0.171 0.296 0.940" mode="trackcom"/>->'
    '<camera name="front" pos="0.4 0 0.15" xyaxes="0 -1 0 0 0 1" fovy="47.5"/>;'
    "<worldbody>-><worldbody>"
    '<light pos="0 0 4" dir="0.3 0.2 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>'
)
try:
    import numpy as np
    import torch

    assert torch.cuda.is_available()
except Exception as e:
    print("skip: torch+cuda unavailable:", e)
    sys.exit(77)
import _mjmlx_rl_native as native
from mjmlx_camera_cuda import CudaCamera
from mjmlx_go2_env import MjmlxGo2Env

env = MjmlxGo2Env(native, 8, cmd_vel_min=0.2, cmd_vel_max=1.0, use_gpu=True, seed=0, terrain_seed=0, flat_terrain=True)
env.reset()
cam = native.create_camera(env._sim, 64, 48)
g = CudaCamera.try_create(cam)
if g is None:
    print("skip: CUDA interop unsupported")
    sys.exit(77)
rng = np.random.default_rng(0)
for step in range(3):
    env.step(torch.from_numpy(rng.normal(0, 0.5, (8, 12)).astype(np.float32)))
    cam.render()
    rgba, depth = g.read()
    torch.cuda.synchronize()
    rgb_cpu, depth_cpu = cam.rgb(), cam.depth()
    assert (rgba[..., :3].cpu().numpy() == rgb_cpu).all(), f"rgb mismatch at step {step}"
    assert (depth.cpu().numpy() == depth_cpu).all(), f"depth mismatch at step {step}"
    assert rgb_cpu.std() > 1.0, "image is blank"
print("camera_cuda OK: rgba", tuple(rgba.shape), "depth", tuple(depth.shape))
