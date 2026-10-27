#!/usr/bin/env python3
"""hfield(depth)とテクスチャ(色)をMuJoCo標準レンダラ(mujoco.Renderer, GL要)と比較する。

usage: test_camera_mujoco.py BUILD_DIR [OUT_DIR]
"""

import os
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")
sys.path.insert(0, sys.argv[1])
import mujoco
import numpy as np
import _mjmlx_rl_native as native
from PIL import Image

OUT = sys.argv[2] if len(sys.argv) > 2 else None
W = H = 96
fails = []


def render_both(xml, cam="c", assets=None):
    d = tempfile.mkdtemp()
    for name, arr in (assets or {}).items():
        Image.fromarray(arr).save(os.path.join(d, name))
    path = os.path.join(d, "m.xml")
    open(path, "w").write(xml)
    model = native.load_model(path)
    sim = native.create_batched(model, 1, use_gpu=True)
    c = native.create_camera(sim, W, H)
    c.render()
    rgb, depth = c.rgb()[0, 0], c.depth()[0, 0]
    m = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    r = mujoco.Renderer(m, H, W)
    r.update_scene(data, camera=cam)
    rgb_m = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(data, camera=cam)
    depth_m = r.render().copy()
    return rgb, depth, rgb_m, depth_m, m.vis.map.zfar * m.stat.extent


def check(name, ok, msg):
    print(("PASS " if ok else "FAIL ") + name + ": " + msg)
    if not ok:
        fails.append(name)


def save(name, a, b):
    if OUT:
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(np.concatenate([a, b], axis=1)).save(f"{OUT}/{name}.png")


BODY = '<body pos="0 0 -50"><freejoint/><geom type="sphere" size="0.01" mass="1" group="3"/></body>'
UNLIT = '<visual><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>'

# --- hfield: depth ---
rng = np.random.default_rng(0)
n = 40
yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
elev = 0.5 + 0.3 * np.sin(6 * xx) * np.cos(5 * yy) + 0.1 * rng.random((n, n))
elev = (elev - elev.min()) / (elev.max() - elev.min())
xml = f'''<mujoco><asset><hfield name="h" nrow="{n}" ncol="{n}" size="3 3 0.6 0.1" elevation="{" ".join(f"{v:.4f}" for v in elev.ravel())}"/></asset>
<worldbody><geom type="hfield" hfield="h" rgba="0.6 0.5 0.4 1"/>{BODY}
<camera name="c" pos="0 -5 4" xyaxes="1 0 0 0 0.62 0.78" fovy="45"/></worldbody></mujoco>'''
rgb, depth, rgb_m, depth_m, zfar_m = render_both(xml)
hit_o, hit_m = depth < 99, depth_m < 0.5 * zfar_m
both = hit_o & hit_m
err = np.abs(depth - depth_m)[both]
iou = both.sum() / max((hit_o | hit_m).sum(), 1)
print(f"hfield: hit IoU={iou:.4f} depth |diff| mean={err.mean():.4f} p95={np.percentile(err, 95):.4f}")
check("hfield_mask_iou", iou > 0.97, f"{iou:.4f}")
check("hfield_depth_p95", np.percentile(err, 95) < 0.08, f"{np.percentile(err, 95):.4f} m")
save("hfield", rgb, rgb_m)

# --- texture: color (照明なし=ambientのみで比較) ---
uv = np.zeros((64, 64, 3), np.uint8)
uv[:, :, 0] = np.arange(64)[None, :] * 4
uv[:, :, 1] = np.arange(64)[:, None] * 4
uv[:, :, 2] = 128
cases = {
    "checker_plane": ('<texture name="t" type="2d" builtin="checker" rgb1="0.9 0.2 0.2" rgb2="0.2 0.2 0.9" width="64" height="64"/>', 'texrepeat="3 3"', "plane", "2 2 0.1"),
    "checker_uniform": ('<texture name="t" type="2d" builtin="checker" rgb1="0.9 0.9 0.2" rgb2="0.1 0.5 0.1" width="64" height="64"/>', 'texrepeat="2 2" texuniform="true"', "plane", "3 2 0.1"),
    "png_plane": ('<texture name="t" type="2d" file="uv.png"/>', 'texrepeat="1 1"', "plane", "2 2 0.1"),
    "png_box": ('<texture name="t" type="2d" file="uv.png"/>', 'texrepeat="1 1"', "box", "1 1 0.3"),
}
for name, (tex, matattr, gtype, size) in cases.items():
    xml = f'''<mujoco>{UNLIT}<asset>{tex}<material name="m" texture="t" {matattr} specular="0" shininess="0"/></asset>
<worldbody><geom type="{gtype}" size="{size}" material="m"/>{BODY}
<camera name="c" pos="0 0 5" xyaxes="1 0 0 0 1 0" fovy="50"/></worldbody></mujoco>'''
    rgb, depth, rgb_m, depth_m, _ = render_both(xml, assets={"uv.png": uv})
    mask = (depth < 99) & (depth_m < 99)
    diff = np.abs(rgb.astype(int) - rgb_m.astype(int))[mask].mean()
    print(f"{name}: mean |color diff| = {diff:.2f} / 255 over {mask.sum()} px")
    check(name, diff < 12, f"{diff:.2f}")
    save(name, rgb, rgb_m)

print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
