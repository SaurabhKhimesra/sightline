"""Data the cell carries: the park pose found in gate 1."""
from pathlib import Path

import yaml


def park_pose() -> list:
    """The UR5e's park joint angles (rad), the pose found in gate 1 that hides nothing
    from the overhead camera (results/gate1/gate1.yaml, park_pose)."""
    return yaml.safe_load((Path(__file__).parent / "park_pose.yaml").read_text())["joints_rad"]
