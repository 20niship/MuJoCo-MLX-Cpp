#!/usr/bin/env python3
"""test_gpu_vs_cpu --dump の軌道(CPU/GPUのqpos)をmujocoでレンダリングし、N行x2列(CPU | GPU)のcaption付きタイル動画に書き出す。

usage: render_compare.py OUT_DIR MODEL.xml:TRACE.bin [MODEL.xml:TRACE.bin ...]
出力: OUT_DIR/compare.mp4 と最終フレームの OUT_DIR/compare_last.png (ffmpegが必要)
"""
import os
import subprocess
import sys
from pathlib import Path

# 描画バックエンド: DISPLAYがあればglfw、無ければ(headless)egl。MUJOCO_GL環境変数で上書き可。
os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H = 480, 360


def load_trace(path):
    raw = Path(path).read_bytes()
    nq, n = np.frombuffer(raw[:8], np.int32)
    fps = float(np.frombuffer(raw[8:12], np.float32)[0])
    qpos = np.frombuffer(raw[12:], np.float32).reshape(2, n, nq)  # [0]=CPU, [1]=GPU
    return qpos, fps


class Row:
    def __init__(self, model_path, trace_path):
        parent = Path(model_path).parent
        self.name = parent.parent.name if parent.name.startswith("v") and parent.name[1:2].isdigit() else parent.name
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, TILE_H, TILE_W)
        self.qpos, self.fps = load_trace(trace_path)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        # 床(plane)を除いたロボットのサイズからカメラ距離を決める(stat.extentは床を含み遠すぎる)。注視点はロボットのgeom重心に追従。
        mujoco.mj_forward(self.model, self.data)
        self.body_geoms = self.model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        pts = self.data.geom_xpos[self.body_geoms]
        self.cam.distance = 2.0 * max(float(np.ptp(pts, axis=0).max()), 0.3)
        self.cam.azimuth, self.cam.elevation = 135, -20

    def render(self, col, frame):
        self.data.qpos[:] = self.qpos[col, frame]
        mujoco.mj_forward(self.model, self.data)
        self.cam.lookat[:] = self.data.geom_xpos[self.body_geoms].mean(axis=0)
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()


def caption(img, text, font):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=font)
    d.rectangle([0, 0, w + 16, 30], fill=(0, 0, 0))
    d.text((8, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [Row(*spec.rsplit(":", 1)) for spec in sys.argv[2:]]
    nframes = min(r.qpos.shape[1] for r in rows)
    fps = min(r.fps for r in rows)
    font = ImageFont.load_default(size=20)
    labels = ["CPU (MuJoCo C)", "GPU (MKX/Vulkan)"]

    mp4 = out_dir / "compare.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * TILE_W}x{len(rows) * TILE_H}",
         "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        stdin=subprocess.PIPE)
    last = None
    for f in range(nframes):
        tiles = [[caption(r.render(c, f), f"{r.name} | {labels[c]} | t={f / fps:.2f}s", font) for c in range(2)] for r in rows]
        last = np.vstack([np.hstack(t) for t in tiles])
        ff.stdin.write(last.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    Image.fromarray(last).save(out_dir / "compare_last.png")
    print(f"wrote {mp4} ({nframes} frames, {fps:g} fps, {len(rows)}x2 tiles)")


if __name__ == "__main__":
    main()
