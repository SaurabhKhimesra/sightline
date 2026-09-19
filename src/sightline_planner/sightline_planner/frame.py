"""What the planner is allowed to see, and nothing else.

SPEC.md section 8.1: eyes camera colour and depth, wrist camera colour, its own
joint positions and velocities, the static station model, its own robot model, the
product model (holes relative to the part), and the task state it keeps itself.

It never gets human geometry, segmentation, the true box pose, or the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorFrame:
    """One tick of what the cell tells the robot."""
    t: float
    q: np.ndarray                      # joint positions, rad
    qd: np.ndarray                     # joint velocities, rad/s
    wrist_rgb: np.ndarray | None = None   # 640 x 480 colour, 25 Hz
    eyes_rgb: np.ndarray | None = None    # 640 x 480 colour, 25 Hz, one frame late
    eyes_depth: np.ndarray | None = None  # metres, same frame
    # what a jig switch or a button tells the cell. Not a look at the person.
    jig_signal: str = "empty"          # empty, rail_ready, cover_on, cover_ready, done: what the jig switches say


@dataclass
class Taught:
    """Static cell knowledge: where the jig holds the part and where the screws are.

    The jig is bolted to the bench, so its nominal part pose belongs to the station
    model. Hole positions are given relative to the part, as SPEC.md section 8.1
    allows. Any difference between the nominal pose and the real one is what the
    wrist camera is for.
    """
    part_pose: tuple                   # x, y, z of the part's origin in the cell
    rail_holes: list = field(default_factory=list)   # relative to the part origin
    cover_holes: list = field(default_factory=list)
    feeder_pick: tuple = (0.0, 0.0, 0.0)
    screw_length: float = 0.0156
    approach: float = 0.06             # stand-off above a hole, m (scene choice)
    cover_size: tuple = (0.200, 0.150)  # the part's own drawing, not its pose
    cover_top: float = 0.086            # top face above the part origin, from the drawing
    rail_origin_dz: float = 0.021       # the rail sits on its bosses, from the drawing
    cover_origin_dz: float = 0.082      # the cover's own origin, 4 mm under its top face

    def world(self, hole) -> np.ndarray:
        return np.asarray(self.part_pose, float) + np.asarray(hole, float)
