#!/usr/bin/env python3
"""mesh付き実ロボットMJCFをbenchmarks/models/external/へ取得する(.gitignore済み = git管理外)。取得済みならskip。"""
import subprocess
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parent.parent / "benchmarks" / "models" / "external"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_DIRS = ["agilex_piper", "unitree_go2", "unitree_g1", "unitree_h1"]
OPENARM_URL = "https://github.com/enactic/openarm_mujoco.git"


def run(*args):
    subprocess.run(args, check=True)


def fetch_menagerie():
    dst = EXTERNAL / "mujoco_menagerie"
    if not (dst / ".git").exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE_URL, str(dst))
    missing = [d for d in MENAGERIE_DIRS if not (dst / d).exists()]
    if missing:
        print(f"fetching mujoco_menagerie: {missing}")
        run("git", "-C", str(dst), "sparse-checkout", "add", *missing)


def fetch_openarm():
    dst = EXTERNAL / "openarm_mujoco"
    if not dst.exists():
        print("fetching openarm_mujoco")
        run("git", "clone", "--depth", "1", OPENARM_URL, str(dst))


if __name__ == "__main__":
    fetch_menagerie()
    fetch_openarm()
