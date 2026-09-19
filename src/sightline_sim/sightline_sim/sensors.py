"""The two cameras, as the planner gets to see them.

The eyes camera is a RealSense D455 stand-in: 640 x 480 colour and depth at 25 Hz,
one frame late, with depth noise and missing depth at edges. The wrist camera is a
D405 stand-in: colour only.

Depth noise follows Nguyen, Izadi and Lovell 2012, as SPEC.md section 8.2 sets out:
axial sigma_z = 0.0012 + 0.0019 (z - 0.4)^2 m (their Eq. 3, for surface angles 10 to
60 degrees), lateral sigma_L = 0.8 + 0.035 theta / (pi/2 - theta) px (their Eq. 1).
It was fitted on a Kinect between 0.5 and 2.75 m and is used here as a stand-in for
an RGB-D camera. Everything else here is a scene choice.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from . import station

EDGE_STEP_M = 0.02      # a depth jump this big across a pixel loses the return (scene choice)
MAX_RANGE_M = 4.0       # beyond this the depth image is empty (scene choice)
# Nguyen et al. fitted their noise between 0.5 and 2.75 m. Extrapolated to the far
# side of a hall it gives 60 mm of noise at 6 m, which swamped the background test
# and turned half the floor into "maybe the person". It is held flat past the fit.
FIT_RANGE_M = (0.4, 2.75)


def _visible_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (station.GROUP_ENV, station.GROUP_WORKER, station.GROUP_ROBOT) else 0
    return opt


class Camera:
    """A colour camera on the simulated cell."""

    def __init__(self, model: mujoco.MjModel, name: str, width: int = 640, height: int = 480):
        self.m = model
        self.name = name
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        self.opt = _visible_option()

    def rgb(self, d: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(d, camera=self.name, scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


class DepthCamera:
    """Depth in metres along the camera axis, with noise, dropouts and a delay."""

    def __init__(self, model: mujoco.MjModel, name: str = "eyes_cam", width: int = 640,
                 height: int = 480, seed: int = 0, delay_frames: int = 1):
        self.m = model
        self.name = name
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        self.renderer.enable_depth_rendering()
        self.opt = _visible_option()
        self.rng = np.random.default_rng(seed)
        self.queue: deque = deque(maxlen=max(delay_frames, 0) + 1)
        self.delay_frames = delay_frames
        self.vv, self.uu = np.meshgrid(np.arange(height, dtype=np.int32),
                                       np.arange(width, dtype=np.int32), indexing="ij")

    def raw(self, d: mujoco.MjData, groups=None) -> np.ndarray:
        """Perfect depth. groups limits what is in the picture, for a background map."""
        opt = self.opt
        if groups is not None:
            opt = mujoco.MjvOption()
            for g in range(6):
                opt.geomgroup[g] = 1 if g in groups else 0
        self.renderer.update_scene(d, camera=self.name, scene_option=opt)
        return np.asarray(self.renderer.render(), dtype=np.float64)

    def spoil(self, depth: np.ndarray) -> np.ndarray:
        """Add what a real depth camera does to a perfect depth image.

        Float32 and one three deep noise draw, because this runs on every frame of
        every episode and the arithmetic is the whole cost.
        """
        z = np.asarray(depth, np.float32)
        far = z > MAX_RANGE_M
        gy, gx = np.gradient(z)                         # surface angle, for both noise terms
        slope = np.hypot(gx, gy)
        step = slope > EDGE_STEP_M                      # depth discontinuity: no return
        theta = np.clip(np.arctan(slope * self.pixel_scale(z)), 0.0, np.radians(80.0))
        noise = self.rng.standard_normal((3,) + z.shape, dtype=np.float32)
        z = z + noise[0] * axial_sigma(z).astype(np.float32)          # Nguyen et al. Eq. 3
        # lateral noise displaces where a sample lands, in pixels (their Eq. 1)
        sigma_l = np.clip(0.8 + 0.035 * theta / np.maximum(np.pi / 2 - theta, 1e-3), 0.0, 6.0)
        du = np.rint(noise[1] * sigma_l).astype(np.int32)
        dv = np.rint(noise[2] * sigma_l).astype(np.int32)
        z = z[np.clip(self.vv + dv, 0, z.shape[0] - 1), np.clip(self.uu + du, 0, z.shape[1] - 1)]
        z[step | far] = 0.0                             # 0 means no depth here
        return z.astype(np.float64)

    def pixel_scale(self, z: np.ndarray) -> np.ndarray:
        """Metres per pixel at each depth, for turning a depth gradient into an angle."""
        f = self.focal()
        return np.maximum(z, 1e-6) / f

    def focal(self) -> float:
        fovy = float(self.m.cam_fovy[self.m.cam(self.name).id])
        return (self.height / 2) / np.tan(np.radians(fovy) / 2)

    def intrinsics(self) -> tuple[float, float, float]:
        """Focal length in pixels and the principal point, pixel centre convention.

        Unprojecting with u - width/2 instead of u + 0.5 - width/2 puts every point
        4 mm out at 2.5 m. With the half pixel it is 1.6 mm, which is the floor of
        this depth path and well under the noise the camera itself adds.
        """
        return self.focal(), self.width / 2 - 0.5, self.height / 2 - 0.5

    def capture(self, d: mujoco.MjData) -> np.ndarray:
        """The depth image the planner gets now: this frame's, delayed and spoiled."""
        self.queue.append(self.spoil(self.raw(d)))
        return self.queue[0] if len(self.queue) > self.delay_frames else np.zeros((self.height, self.width))

    def close(self) -> None:
        self.renderer.close()


def empty_station_depth(cam: "DepthCamera", m: mujoco.MjModel, d: mujoco.MjData,
                        parts=("part_base", "part_rail", "part_cover")) -> np.ndarray:
    """The reference depth map: the cell with nobody in it and nothing in the jig.

    A real cell takes this once, with the jig empty. Here the parts are lifted out of
    the picture and put back. Leaving them in makes the box part of the background,
    and then the box pose fit has only the worker's hands to look at.
    """
    keep = {}
    for part in parts:
        try:
            adr = m.joint(part + "_free").qposadr[0]
        except KeyError:
            continue
        keep[part] = d.qpos[adr:adr + 7].copy()
        d.qpos[adr:adr + 3] = [0.0, 0.0, -20.0]
    mujoco.mj_forward(m, d)
    depth = cam.raw(d, groups=(station.GROUP_ENV,))
    for part, pose7 in keep.items():
        adr = m.joint(part + "_free").qposadr[0]
        d.qpos[adr:adr + 7] = pose7
    mujoco.mj_forward(m, d)
    return depth


def axial_sigma(z) -> np.ndarray:
    """Depth noise, one standard deviation, in metres (Nguyen et al. 2012, Eq. 3)."""
    zz = np.clip(np.asarray(z, float), *FIT_RANGE_M)
    return 0.0012 + 0.0019 * (zz - 0.4) ** 2


def camera_pose(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Where a camera is and which way it looks, in the world frame.

    Camera frames are computed by mj_camlight, not by mj_kinematics, so a caller that
    has only moved joints gets the origin and a zero rotation, and every point it
    unprojects lands at (0, 0, 0). This calls it, and refuses to return the zero pose.
    """
    mujoco.mj_camlight(m, d)
    cam = m.cam(name).id
    pos, R = d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy()
    if not np.any(R):
        raise RuntimeError(f"camera {name} has no pose yet: call mj_forward first")
    return pos, R
