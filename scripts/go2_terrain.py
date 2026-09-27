"""Go2用の階段状ボックス地形(IsaacLab pyramid_stairsの1D簡易版)を生成する。"""

from __future__ import annotations

import numpy as np

CORRIDOR_WIDTH = 3.0  # 通路の全幅[m]
STEP_HEIGHT_RANGE = (0.03, 0.05)  # 段差の高さ[m]
STEP_DEPTH_RANGE = (0.20, 0.50)  # 段の奥行き(進行方向の幅)[m]
NUM_STEPS_UP = 10  # 片側の段数(上り。下りは折り返しで同数)
FLAT_MARGIN = 1.5  # 地形の両端に付ける平坦部の奥行き[m]


class StairsTerrain:
    """階段の box 形状リストと、x座標から段の天面高さを引く関数を持つ。"""

    def __init__(self, seed: int | None = None):
        rng = np.random.default_rng(seed)

        depths = rng.uniform(*STEP_DEPTH_RANGE, size=NUM_STEPS_UP)
        heights = np.cumsum(rng.uniform(*STEP_HEIGHT_RANGE, size=NUM_STEPS_UP))

        # edges[i]..edges[i+1] が区間iの境界、tops[i] がその区間の天面高さ(平坦->上り->頂上平坦->下り->平坦)
        edges = [0.0]
        tops = []
        edges.append(edges[-1] + FLAT_MARGIN)
        tops.append(0.0)
        for d, h in zip(depths, heights):
            edges.append(edges[-1] + d)
            tops.append(h)
        top_height = heights[-1]
        edges.append(edges[-1] + FLAT_MARGIN)
        tops.append(top_height)
        for d, h in zip(reversed(depths), reversed(heights[:-1].tolist() + [0.0])):
            edges.append(edges[-1] + d)
            tops.append(h)
        edges.append(edges[-1] + FLAT_MARGIN)
        tops.append(0.0)

        self.edges = np.array(edges)
        self.tops = np.array(tops)
        self.total_length = self.edges[-1]

    def height_at(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, self.total_length)
        idx = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.tops) - 1)
        return self.tops[idx]

    def sample_spawn_xy(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        margin = 0.3
        edge_margin = 0.08  # 段差の継ぎ目ちょうどに立たせると足がすり抜けることがある(実測で確認)ので避ける
        x = np.empty(n)
        need = np.ones(n, dtype=bool)
        while need.any():
            cand = rng.uniform(margin, self.total_length - margin, size=need.sum())
            near_edge = np.min(np.abs(cand[:, None] - self.edges[None, :]), axis=1) < edge_margin
            idx = np.nonzero(need)[0]
            x[idx[~near_edge]] = cand[~near_edge]
            need[idx[~near_edge]] = False
        y = rng.uniform(-CORRIDOR_WIDTH / 2 + margin, CORRIDOR_WIDTH / 2 - margin, size=n)
        return x, y

    def to_mjcf_bodies(self) -> str:
        """区間ごとのboxをworldbody直下のgeomとして書き出す(高さ0の区間は素通りできる薄い床にする)。"""
        parts = []
        for i in range(len(self.tops)):
            x0, x1 = self.edges[i], self.edges[i + 1]
            top = max(self.tops[i], 0.005)
            cx = (x0 + x1) / 2.0
            hx = (x1 - x0) / 2.0
            parts.append(
                f'<geom type="box" pos="{cx:.4f} 0 {top / 2:.4f}" '
                f'size="{hx:.4f} {CORRIDOR_WIDTH / 2:.4f} {top / 2:.4f}" '
                f'friction="0.8 0.02 0.01" rgba="0.5 0.5 0.55 1"/>'
            )
        return "\n    ".join(parts)
