"""Draw the two candidate sight lines in the scene, so the camera choice is visible.

Red line: a camera in front of the cell, above the worker, aimed at the box.
Green line: the pole beside the bench, aimed at the same point.

Usage: MUJOCO_GL=egl python scripts/eyes_explain.py [out_dir]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sightline_sim import geometry as G, station  # noqa: E402
from scripts.gate1_stills import pose_scene  # noqa: E402


def build_with_lines(layout: station.Layout):
    """The station plus two marked sight lines and two extra viewpoints."""
    scene_spec_builder = station.build
    scene = scene_spec_builder(layout)  # first build to learn the worktop height
    zw = scene.worktop_z
    target = np.array([0.0, -0.06, zw + 0.04])
    front = np.array([0.0, -1.22, zw + 1.80])
    pole = np.array([1.15, -0.90, zw + 2.13])
    del scene

    original = station.build_eyes_camera

    def patched(b, L, zwl, frame_info):
        out = original(b, L, zwl, frame_info)
        body = b.spec.worldbody.add_body(name="sightlines")
        b.material("diag_red", rgba=[0.88, 0.10, 0.08, 1], emission=0.35, specular=0.2)
        b.material("diag_green", rgba=[0.10, 0.72, 0.28, 1], emission=0.35, specular=0.2)
        for name, cam, mat in (("front", front, "diag_red"), ("pole", pole, "diag_green")):
            b.capsule(body, 0.008, cam, target, mat)
            b.sphere(body, 0.05, cam, mat)
        # two viewpoints that show both lines next to the worker's head
        wb = b.spec.worldbody
        for cname, pos, look, fovy, up in (
                ("explain_side", (-3.70, -0.52, 1.72), (0.38, -0.46, 1.44), 33.0, (0.0, 0.0, 1.0)),
                ("explain_top", (0.05, -0.65, 5.4), (0.05, -0.45, 1.2), 34.0, (0.0, 1.0, 0.0))):
            x, y = np.split(np.array(G.look_at_xyaxes(pos, look, up)), 2)
            R = np.column_stack([x, y, np.cross(x, y)])
            wb.add_camera(name=cname, pos=list(pos), quat=G.mat_to_quat(R), fovy=fovy)
        return out

    station.build_eyes_camera = patched
    try:
        return station.build(layout)
    finally:
        station.build_eyes_camera = original


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate1/eyes_explain")
    out.mkdir(parents=True, exist_ok=True)
    scene = build_with_lines(station.Layout())
    d = mujoco.MjData(scene.model)
    pose_scene(scene, d)
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
    for cam, (w, h) in (("explain_side", (1600, 900)), ("explain_top", (1280, 960))):
        with mujoco.Renderer(scene.model, h, w) as r:
            r.update_scene(d, camera=cam, scene_option=opt)
            imageio.imwrite(out / f"{cam}.png", r.render())
    print("wrote", out)


if __name__ == "__main__":
    main()
