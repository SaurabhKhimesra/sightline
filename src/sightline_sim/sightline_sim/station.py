"""The screwdriving station and the hall around it, built with mujoco.MjSpec.

World frame: z up, origin on the floor under the bench centre, +y from the
worker toward the robot, +x to the worker's right.

Numbers carry their source. Everything else is a scene choice, picked to look
like a real assembly cell; docs/notes.md lists them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import mujoco
import numpy as np

from . import geometry as G
from . import textures as T
from .worker import HUMAN_PREFIX, attach_worker

GROUP_ENV = 0
GROUP_WORKER = 1
GROUP_ROBOT = 2
GROUP_COLLISION = 3

ROBOT_PREFIX = "ur5e/"
UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# UR5e control box W x H x D and teach pendant W x H x D, metres (UR5e technical
# specification, updated May 2025).
UR_CONTROL_BOX = (0.460, 0.449, 0.254)
UR_TEACH_PENDANT = (0.300, 0.231, 0.050)

# OnRobot Screwdriver, datasheet v1.7, p. 2 and drawing p. 34. Body 308.5 mm to
# the nose end, 322 mm to the bit holder tip, 86 mm wide, 114.1 mm deep; nose
# diameter 49 mm, 60.4 mm below the body; bit holder 13.5 mm long, 13.5 mm across;
# robot flange centre 166.6 mm above the bit holder tip. The flange mounts on the
# back face. SCREW_AXIS_OFFSET is measured off that drawing (about +-2 mm), and
# QUICK_CHANGER_T is a scene choice.
SD_WIDTH = 0.086
SD_DEPTH = 0.1141
SD_BODY_HEIGHT = 0.322 - 0.0739
SD_NOSE_D = 0.049
SD_NOSE_LEN = 0.0604
SD_HOLDER = 0.0135
SD_FLANGE_ABOVE_TIP = 0.1666
QUICK_CHANGER_T = 0.012
SCREW_AXIS_OFFSET = QUICK_CHANGER_T + 0.0815

# Intel RealSense D455 (eyes) and D405 (wrist): depth field of view and housing
# size from Intel's product specification pages. D455: 86 x 57 deg, 124 x 29 x
# 26 mm. D405: 87 x 58 deg, 42 x 42 x 23 mm, ideal range 7 to 50 cm.
D455_FOVY = 57.0
D455_SIZE = (0.124, 0.029, 0.026)
D405_FOVY = 58.0
D405_SIZE = (0.042, 0.042, 0.023)

# the MuJoCo Menagerie checkout (google-deepmind/mujoco_menagerie), for the UR5e model
MENAGERIE = Path(os.environ.get("MUJOCO_MENAGERIE", Path.home() / "mujoco_menagerie"))


@dataclass
class Layout:
    """Scene choices for the station, metres."""
    worktop_x: float = 1.60
    worktop_y: float = 0.80
    board_t: float = 0.030
    mat_t: float = 0.003
    frame_height: float = 1.10  # top frame above the worktop
    profile: float = 0.040
    robot_xy: tuple = (0.0, 0.215)
    robot_plate: tuple = (0.20, 0.20, 0.015)
    jig_xy: tuple = (0.0, -0.100)
    jig_plate: tuple = (0.32, 0.25, 0.012)
    box_size: tuple = (0.200, 0.150, 0.080)
    feeder_xy: tuple = (0.40, 0.19)
    tray_xy: tuple = (0.53, -0.20)
    # 15 cm further from the jig than first placed (was -0.40): at -0.40 his upper arm,
    # clipping blocks there, was 17 to 25 cm from the robot's wrist at the rail screws,
    # inside the planner's margin. Here it is about 32 cm (scene choice)
    prep_xy: tuple = (-0.55, -0.21)
    worker_xy: tuple = (0.02, -0.61)
    eyes_variant: str = "pole_far"  # see EYES_VARIANTS
    with_cover: bool = True  # False shows the step before the cover goes on (rail screws visible)
    # How many cover screws are already in. The cycle starts with none, the robot
    # drives them. The gate 1 stills posed the scene mid work and show two.
    driven_screws: int = 0
    elbow_to_worktop: float = 0.075  # CCOHS: light work 5 to 10 cm below elbow height; middle


@dataclass
class Scene:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    layout: Layout
    worktop_z: float
    worker_scale: float
    points: dict = field(default_factory=dict)


class Builder:
    """Thin helpers over MjSpec for textures, materials, meshes and geoms.

    With assets_dir set, every texture is written as a PNG and every mesh as an OBJ in
    that directory and the model refers to the files. That is what lets the scene be
    saved as an MJCF file other tools can read: Gazebo's mjcf2sdf converter and
    mujoco_ros2_control both take a file, and MJCF cannot hold in-memory textures.
    """

    def __init__(self, spec: mujoco.MjSpec, assets_dir=None):
        self.assets_dir = None if assets_dir is None else Path(assets_dir)
        if self.assets_dir is not None:
            self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self._n = 0

    def texture(self, name: str, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        if self.assets_dir is not None:
            import imageio.v2 as imageio
            path = self.assets_dir / f"{name}.png"
            imageio.imwrite(path, np.ascontiguousarray(rgb))
            self.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path.resolve()))
            return
        t = self.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3)
        t.data = np.ascontiguousarray(rgb[::-1]).tobytes()

    def material(self, name: str, texture: str | None = None, **props) -> None:
        mat = self.spec.add_material(name=name)
        if texture:
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
        for key, val in props.items():
            setattr(mat, key, val)

    def mesh(self, mesh: G.Mesh, name: str | None = None) -> str:
        self._n += 1
        name = name or f"m{self._n}"
        me = self.spec.add_mesh(name=name)
        if self.assets_dir is not None:
            me.file = str(write_obj(self.assets_dir / f"{name}.obj", mesh.v, mesh.f, mesh.uv))
        else:
            me.uservert = mesh.v.ravel().tolist()
            me.userface = mesh.f.ravel().tolist()
            if mesh.uv is not None:
                me.usertexcoord = mesh.uv.ravel().tolist()
                me.userfacetexcoord = mesh.f.ravel().tolist()
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        return name

    def _geom(self, body, collide: bool, group: int, material: str | None, rgba, **kw):
        g = body.add_geom(group=group, contype=1 if collide else 0, conaffinity=1 if collide else 0, **kw)
        if material:
            g.material = material
        if rgba is not None:
            g.rgba = rgba
        g.density = 0.0 if not collide else g.density
        return g

    def box(self, body, half, pos, material=None, rot=None, collide=False, group=GROUP_ENV, rgba=None, name=None):
        kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos))
        if rot is not None:
            kw["quat"] = G.mat_to_quat(rot)
        if name:
            kw["name"] = name
        return self._geom(body, collide, group, material, rgba, **kw)

    def cyl(self, body, r, half_h, pos, material=None, rot=None, collide=False, group=GROUP_ENV, rgba=None, name=None):
        kw = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, half_h, 0], pos=list(pos))
        if rot is not None:
            kw["quat"] = G.mat_to_quat(rot)
        if name:
            kw["name"] = name
        return self._geom(body, collide, group, material, rgba, **kw)

    def sphere(self, body, r, pos, material=None, group=GROUP_ENV, rgba=None):
        return self._geom(body, False, group, material, rgba, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], pos=list(pos))

    def capsule(self, body, r, p0, p1, material=None, group=GROUP_ENV, rgba=None):
        return self._geom(body, False, group, material, rgba, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[*p0, *p1], size=[r, 0, 0])

    def meshgeom(self, body, mesh: G.Mesh, material=None, pos=(0, 0, 0), rot=None, group=GROUP_ENV, rgba=None, name=None):
        kw = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=self.mesh(mesh), pos=list(pos))
        if rot is not None:
            kw["quat"] = G.mat_to_quat(rot)
        if name:
            kw["name"] = name
        return self._geom(body, False, group, material, rgba, **kw)

    def quad(self, body, w, h, pos, rot, material, u_repeat=1.0, v_repeat=1.0, group=GROUP_ENV):
        """Textured rectangle in its local x-y plane, facing +z."""
        v = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
        uv = np.array([[0, 0], [u_repeat, 0], [u_repeat, v_repeat], [0, v_repeat]])
        mesh = G.Mesh(v, [[0, 1, 2], [0, 2, 3]], uv)
        return self.meshgeom(body, mesh, material, pos, rot, group)


# ---------------------------------------------------------------- materials

def add_materials(b: Builder) -> None:
    b.texture("t_floor", T.epoxy_floor(1024))
    b.texture("t_hazard", T.hazard_stripes())
    b.texture("t_wall", T.wall_panels())
    b.texture("t_window", T.window())
    b.texture("t_kraft", T.kraft())
    b.texture("t_esd", T.esd_mat())
    b.texture("t_worktop", T.plain_noise((0.66, 0.67, 0.66), 256, 0.03))
    b.texture("t_alu", T.brushed((0.80, 0.82, 0.84)))
    b.texture("t_hmi", T.hmi_screen())
    b.texture("t_sign_glasses", T.sign("SAFETY GLASSES\nREQUIRED", 512, 256, px=6))
    b.texture("t_sign_line", T.sign("ASSEMBLY  LINE 2", 1024, 192, bg=(0.10, 0.12, 0.14), px=10))
    b.texture("t_sign_cell", T.sign("CELL 01  SCREWDRIVING", 1024, 160, bg=(0.93, 0.62, 0.05), fg=(0.05, 0.05, 0.05), px=6))
    b.texture("t_label_base", T.label("BASE 200X150"))
    b.texture("t_label_cover", T.label("COVER"))
    b.texture("t_label_rail", T.label("DIN RAIL 35"))
    b.texture("t_label_tb", T.label("TERMINAL BLK"))
    b.texture("t_pallet", T.plain_noise((0.62, 0.48, 0.30), 256, 0.08, (32, 8, 2)))
    spec = b.spec
    sky = spec.add_texture(name="t_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           rgb1=[0.30, 0.32, 0.34], rgb2=[0.10, 0.11, 0.12], width=512, height=512)
    del sky

    M = b.material
    M("floor", "t_floor", texrepeat=[0.25, 0.25], texuniform=True, reflectance=0.0, specular=0.22, shininess=0.45)
    M("hazard", "t_hazard", specular=0.2, shininess=0.3)
    M("line_yellow", rgba=[0.93, 0.73, 0.06, 1], specular=0.25, shininess=0.3)
    M("line_white", rgba=[0.88, 0.89, 0.88, 1], specular=0.25, shininess=0.3)
    M("wall", "t_wall", texrepeat=[0.25, 0.25], texuniform=True, specular=0.15, shininess=0.2)
    M("window", "t_window", texrepeat=[0.5, 0.8], texuniform=True, emission=0.9, specular=0.0)
    M("column", rgba=[0.55, 0.60, 0.66, 1], specular=0.3, shininess=0.3)
    M("rack_blue", rgba=[0.09, 0.27, 0.56, 1], specular=0.35, shininess=0.4)
    M("rack_orange", rgba=[0.92, 0.42, 0.07, 1], specular=0.35, shininess=0.4)
    M("pallet", "t_pallet", texrepeat=[2, 2], texuniform=True, specular=0.05)
    M("kraft", "t_kraft", texrepeat=[2.5, 2.5], texuniform=True, specular=0.05, shininess=0.1)
    M("stretch", rgba=[0.80, 0.84, 0.86, 0.55], specular=0.8, shininess=0.9)
    M("alu", "t_alu", texrepeat=[6, 6], texuniform=True, specular=0.55, shininess=0.55)
    M("alu_dark", rgba=[0.42, 0.44, 0.47, 1], specular=0.5, shininess=0.5)
    M("steel", rgba=[0.56, 0.58, 0.61, 1], specular=0.6, shininess=0.6)
    M("steel_dark", rgba=[0.22, 0.23, 0.25, 1], specular=0.4, shininess=0.4)
    M("zinc", rgba=[0.80, 0.82, 0.84, 1], specular=0.9, shininess=0.85)
    M("worktop", "t_worktop", texrepeat=[4, 4], texuniform=True, specular=0.2, shininess=0.3)
    M("worktop_edge", rgba=[0.16, 0.17, 0.18, 1], specular=0.3, shininess=0.3)
    M("esd", "t_esd", texrepeat=[3, 3], texuniform=True, specular=0.08, shininess=0.1)
    M("black", rgba=[0.05, 0.05, 0.06, 1], specular=0.3, shininess=0.4)
    M("rubber", rgba=[0.09, 0.09, 0.10, 1], specular=0.1, shininess=0.1)
    M("grey_plastic", rgba=[0.36, 0.38, 0.41, 1], specular=0.3, shininess=0.35)
    M("light_plastic", rgba=[0.86, 0.87, 0.86, 1], specular=0.3, shininess=0.35)
    M("enclosure", rgba=[0.70, 0.71, 0.68, 1], specular=0.22, shininess=0.45)
    M("hole", rgba=[0.015, 0.015, 0.015, 1], specular=0.0, shininess=0.0)
    M("terminal_grey", rgba=[0.50, 0.52, 0.54, 1], specular=0.25, shininess=0.3)
    M("terminal_pe", rgba=[0.20, 0.55, 0.20, 1], specular=0.25, shininess=0.3)
    M("tote_blue", rgba=[0.10, 0.32, 0.66, 1], specular=0.3, shininess=0.35)
    M("tote_grey", rgba=[0.40, 0.42, 0.44, 1], specular=0.3, shininess=0.35)
    M("tote_red", rgba=[0.75, 0.12, 0.10, 1], specular=0.3, shininess=0.35)
    M("tote_yellow", rgba=[0.93, 0.70, 0.08, 1], specular=0.3, shininess=0.35)
    M("grip_red", rgba=[0.78, 0.07, 0.05, 1], specular=0.4, shininess=0.5)
    M("led", rgba=[1.0, 0.97, 0.90, 1], emission=1.0, specular=0.0)
    M("stack_green", rgba=[0.15, 0.95, 0.30, 1], emission=0.85, specular=0.5)
    M("stack_amber_off", rgba=[0.45, 0.30, 0.05, 1], specular=0.6, shininess=0.7)
    M("stack_red_off", rgba=[0.40, 0.06, 0.05, 1], specular=0.6, shininess=0.7)
    M("screen", "t_hmi", emission=0.85, specular=0.2)
    M("glass", rgba=[0.02, 0.02, 0.03, 1], specular=1.0, shininess=1.0)
    M("sign_glasses", "t_sign_glasses", specular=0.2)
    M("sign_line", "t_sign_line", specular=0.1)
    M("sign_cell", "t_sign_cell", specular=0.2)
    for key in ("base", "cover", "rail", "tb"):
        M(f"label_{key}", f"t_label_{key}", specular=0.1)
    M("controller", rgba=[0.33, 0.35, 0.38, 1], specular=0.3, shininess=0.3)
    M("pendant", rgba=[0.17, 0.18, 0.20, 1], specular=0.35, shininess=0.4)
    M("tool_dark", rgba=[0.16, 0.17, 0.19, 1], specular=0.35, shininess=0.45)
    M("tool_light", rgba=[0.66, 0.68, 0.70, 1], specular=0.40, shininess=0.5)
    M("wood", rgba=[0.55, 0.42, 0.28, 1], specular=0.1, shininess=0.2)
    M("fire_red", rgba=[0.80, 0.07, 0.06, 1], specular=0.4, shininess=0.5)
    M("cable", rgba=[0.08, 0.08, 0.09, 1], specular=0.3, shininess=0.4)
    M("monitor_black", rgba=[0.03, 0.03, 0.035, 1], specular=0.5, shininess=0.6)


# ---------------------------------------------------------------- profiles

def tslot_profile(size: float = 0.040) -> np.ndarray:
    """40 mm aluminium profile outline with a T-slot on each side (scene choice)."""
    s = size * 1000.0
    ch = 2.0
    slot = [(-4.0, -s / 2), (-4.0, -s / 2 + 2.2), (-7.0, -s / 2 + 2.2), (-7.0, -s / 2 + 6.5),
            (7.0, -s / 2 + 6.5), (7.0, -s / 2 + 2.2), (4.0, -s / 2 + 2.2), (4.0, -s / 2)]
    pts = []
    for k in range(4):
        c, sn = round(np.cos(k * np.pi / 2)), round(np.sin(k * np.pi / 2))
        R = np.array([[c, -sn], [sn, c]])
        pts += [tuple(R @ np.array(p, float)) for p in [(-s / 2 + ch, -s / 2)] + slot + [(s / 2 - ch, -s / 2)]]
    return np.array(pts) / 1000.0


_PROFILE_CACHE: dict = {}


def profile_bar(b: Builder, body, p0, p1, size: float = 0.040, material: str = "alu", collide: bool = True):
    """Aluminium profile between two points, any direction, with a collision box."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return
    key = (round(length, 5), size)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = G.prism(tslot_profile(size), -length / 2, length / 2, sharp_deg=40)
    mesh = _PROFILE_CACHE[key]
    rot = G.frame_from_axis(d, hint=(0.0, 0.0, 1.0) if abs(d[2]) < 0.9 * length else (1.0, 0.0, 0.0))
    b.meshgeom(body, mesh, material, (p0 + p1) / 2, rot)
    if collide:
        g = b.box(body, (size / 2, size / 2, length / 2), (p0 + p1) / 2, rot=rot, collide=True, group=GROUP_COLLISION)
        g.rgba = [0.5, 0.5, 0.5, 0.3]


# ---------------------------------------------------------------- hall

def build_hall(b: Builder) -> None:
    wb = b.spec.worldbody
    b._geom(wb, True, GROUP_ENV, "floor", None, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.05], name="floor")
    hall = wb.add_body(name="hall")
    # walkway lines and cell floor markings
    for y in (-2.35, -4.05):
        b.box(hall, (7.5, 0.05, 0.001), (0.0, y, 0.001), "line_yellow")
    for x in (-2.75, 2.75):
        b.box(hall, (0.05, 3.35, 0.001), (x, 1.0, 0.001), "line_yellow")
    # hazard tape around the shared cell area, open at the worker's side
    tape_w = 0.05
    xa, xb, ya, yb = -1.45, 1.45, -1.05, 1.10
    for (x0, y0, x1, y1) in ((xa, yb, xb, yb), (xa, ya, xa, yb), (xb, ya, xb, yb),
                             (xa, ya, -0.55, ya), (0.55, ya, xb, ya)):
        length = np.hypot(x1 - x0, y1 - y0)
        rot = G.rot_z(np.arctan2(y1 - y0, x1 - x0))
        b.quad(hall, length, tape_w, ((x0 + x1) / 2, (y0 + y1) / 2, 0.0022), rot, "hazard", u_repeat=length / 0.12 / 8)
    # floor stencil in front of the cell
    b.quad(hall, 1.10, 0.17, (0.0, -1.35, 0.0022), np.eye(3), "sign_cell")

    # back wall with high windows, a roller door and a line sign
    wall_y = 6.0
    b.box(hall, (14.0, 0.10, 4.0), (0.0, wall_y + 0.10, 4.0), "wall")
    b.box(hall, (12.5, 0.02, 0.55), (0.0, wall_y - 0.01, 5.35), "window")
    b.box(hall, (1.6, 0.06, 2.05), (5.0, wall_y - 0.05, 2.05), "steel_dark")
    for k in range(20):
        b.box(hall, (1.58, 0.012, 0.012), (5.0, wall_y - 0.12, 0.2 + k * 0.2), "steel")
    for x in (3.2, 6.8):
        b.cyl(hall, 0.08, 0.55, (x, wall_y - 0.6, 0.55), "line_yellow")
    b.quad(hall, 4.2, 0.79, (-3.0, wall_y - 0.02, 3.4), G.rot_x(np.pi / 2), "sign_line")

    # columns with base guards and a safety sign near the cell
    for (x, y) in ((-4.8, 3.2), (4.8, 3.2), (-4.8, -5.2), (4.8, -5.2)):
        b.box(hall, (0.15, 0.15, 4.0), (x, y, 4.0), "column")
        b.box(hall, (0.22, 0.22, 0.5), (x, y, 0.5), "hazard")
    b.quad(hall, 0.50, 0.25, (-4.8, 3.2 - 0.152, 1.75), G.rot_x(np.pi / 2), "sign_glasses")

    # pallet racking along the back wall
    for bay in range(3):
        x0 = -9.2 + bay * 2.8
        for x in (x0, x0 + 2.7):
            for y in (4.6, 5.6):
                b.box(hall, (0.045, 0.035, 2.6), (x, y, 2.6), "rack_blue")
        for z in (1.55, 3.15, 4.75):
            for y in (4.6, 5.6):
                b.box(hall, (1.35, 0.03, 0.06), (x0 + 1.35, y, z), "rack_orange")
        for lvl, z in enumerate((0.0, 1.61, 3.21)):
            for p in range(2):
                px = x0 + 0.70 + p * 1.3
                b.box(hall, (0.58, 0.45, 0.07), (px, 5.1, z + 0.07), "pallet")
                if (bay + lvl + p) % 3 != 2:
                    h = 0.45 + 0.1 * ((bay + p) % 2)
                    b.box(hall, (0.55, 0.42, h), (px, 5.1, z + 0.14 + h), "kraft")
                else:
                    for i in range(2):
                        for j in range(2):
                            b.box(hall, (0.26, 0.2, 0.18), (px - 0.27 + i * 0.54, 5.1 - 0.21 + j * 0.42, z + 0.32), "tote_blue")

    # neighbouring stations (no robot), so the cell sits in a working line
    for x in (-3.6, 3.6):
        _plain_station(b, hall, x)
    # a parts cart beside the cell
    _cart(b, hall, (-1.85, -0.65))
    # overhead LED high bays and a cable tray
    for x in (-6, -2, 2, 6):
        for y in (-3.0, 1.5):
            b.box(hall, (0.6, 0.12, 0.03), (x, y, 7.2), "steel_dark")
            b.box(hall, (0.58, 0.10, 0.005), (x, y, 7.165), "led")
    b.box(hall, (10.0, 0.15, 0.01), (0.0, 2.2, 4.2), "steel")
    for side in (-1, 1):
        b.box(hall, (10.0, 0.005, 0.04), (0.0, 2.2 + side * 0.15, 4.23), "steel")


def _plain_station(b: Builder, hall, x: float) -> None:
    z = 0.95
    b.box(hall, (0.8, 0.4, 0.015), (x, 0.3, z), "worktop")
    for sx in (-0.78, 0.78):
        for sy in (-0.08, 0.68):
            b.box(hall, (0.02, 0.02, z / 2), (x + sx, sy, z / 2), "alu")
            b.box(hall, (0.02, 0.02, 0.55), (x + sx, sy, z + 0.55), "alu")
    b.box(hall, (0.8, 0.02, 0.02), (x, 0.68, z + 1.1), "alu")
    b.box(hall, (0.8, 0.02, 0.02), (x, -0.08, z + 1.1), "alu")
    b.box(hall, (0.6, 0.03, 0.015), (x, 0.3, z + 1.07), "steel_dark")
    b.box(hall, (0.55, 0.02, 0.005), (x, 0.3, z + 1.052), "led")
    for i, mat in enumerate(("tote_blue", "tote_red", "tote_blue", "tote_yellow")):
        b.box(hall, (0.09, 0.14, 0.06), (x - 0.55 + i * 0.2, 0.55, z + 0.075), mat)
    b.box(hall, (0.25, 0.015, 0.16), (x + 0.45, 0.62, z + 0.35), "monitor_black")
    b.box(hall, (0.23, 0.002, 0.14), (x + 0.45, 0.604, z + 0.35), "screen")
    b.box(hall, (0.2, 0.15, 0.04), (x - 0.1, 0.25, z + 0.055), "enclosure")


def _cart(b: Builder, hall, xy) -> None:
    x, y = xy
    for sx in (-0.33, 0.33):
        for sy in (-0.23, 0.23):
            b.box(hall, (0.012, 0.012, 0.45), (x + sx, y + sy, 0.55), "steel")
            b.cyl(hall, 0.045, 0.012, (x + sx, y + sy, 0.05), "rubber", rot=G.rot_x(np.pi / 2))
    for z in (0.22, 0.62, 0.98):
        b.box(hall, (0.35, 0.25, 0.012), (x, y, z), "grey_plastic")
    for i, mat in enumerate(("tote_blue", "tote_blue")):
        b.box(hall, (0.15, 0.2, 0.08), (x - 0.17 + i * 0.34, y, 0.72), mat)
    b.box(hall, (0.28, 0.21, 0.09), (x, y, 0.32), "kraft")


# ---------------------------------------------------------------- bench and frame

def build_bench(b: Builder, L: Layout, zw: float) -> dict:
    """Bench, lower shelf, UR control box and the top frame. zw is the work surface height."""
    body = b.spec.worldbody.add_body(name="bench")
    hx, hy, p = L.worktop_x / 2, L.worktop_y / 2, L.profile
    board_top = zw - L.mat_t
    board_mid = board_top - L.board_t / 2
    rail_z = board_top - L.board_t - p / 2
    # worktop board with a dark edge band and the ESD mat
    b.box(body, (hx, hy, L.board_t / 2 - 0.001), (0, 0, board_mid), "worktop", collide=True, name="worktop")
    b.box(body, (hx + 0.002, hy + 0.002, 0.004), (0, 0, board_top - 0.005), "worktop_edge")
    b.box(body, (hx - 0.05, hy - 0.04, L.mat_t / 2), (0, 0, board_top + L.mat_t / 2), "esd")
    b.cyl(body, 0.006, 0.002, (hx - 0.07, -hy + 0.06, zw + 0.0005), "zinc")
    # legs with levelling feet, under-rails and a lower shelf
    xs, ys = (-hx + p / 2, hx - p / 2), (-hy + p / 2, hy - p / 2)
    for x in xs:
        for y in ys:
            profile_bar(b, body, (x, y, 0.06), (x, y, board_top - L.board_t), p)
            b.cyl(body, 0.030, 0.006, (x, y, 0.012), "rubber")
            b.cyl(body, 0.008, 0.02, (x, y, 0.038), "zinc")
    for y in ys:
        profile_bar(b, body, (xs[0] + p / 2, y, rail_z), (xs[1] - p / 2, y, rail_z), p)
        profile_bar(b, body, (xs[0] + p / 2, y, 0.20), (xs[1] - p / 2, y, 0.20), p)
    for x in xs:
        profile_bar(b, body, (x, ys[0] + p / 2, rail_z), (x, ys[1] - p / 2, rail_z), p)
        profile_bar(b, body, (x, ys[0] + p / 2, 0.20), (x, ys[1] - p / 2, 0.20), p)
    b.box(body, (hx - p, hy - p, 0.009), (0, 0, 0.229), "grey_plastic")
    # UR control box on the shelf, and a carton of spare bases
    cw, ch, cd = UR_CONTROL_BOX
    cb = (0.30, 0.10, 0.238 + ch / 2)
    b.box(body, (cw / 2, cd / 2, ch / 2), cb, "controller")
    b.box(body, (cw / 2 - 0.02, 0.002, ch / 2 - 0.03), (cb[0], cb[1] - cd / 2 - 0.001, cb[2]), "steel_dark")
    b.cyl(body, 0.03, 0.01, (cb[0] - 0.15, cb[1] - cd / 2 - 0.008, cb[2] + 0.12), "grip_red", rot=G.rot_x(np.pi / 2))
    b.box(body, (0.20, 0.15, 0.12), (-0.40, 0.05, 0.358), "kraft")
    # top frame: four uprights and beams around the top
    top = zw + L.frame_height
    for x in xs:
        for y in ys:
            profile_bar(b, body, (x, y, zw), (x, y, top), p)
    for y in ys:
        profile_bar(b, body, (xs[0] + p / 2, y, top - p / 2), (xs[1] - p / 2, y, top - p / 2), p)
    for x in xs:
        profile_bar(b, body, (x, ys[0] + p / 2, top - p / 2), (x, ys[1] - p / 2, top - p / 2), p)
    beam_y = -0.02
    # two LED light bars along the sides, outside the eyes camera's view
    for x in (xs[0] + 0.14, xs[1] - 0.14):
        b.box(body, (0.030, 0.30, 0.016), (x, 0.0, top - 0.030), "alu_dark")
        b.box(body, (0.024, 0.29, 0.002), (x, 0.0, top - 0.047), "led")
        profile_bar(b, body, (x, ys[0] + p / 2, top - p / 2), (x, ys[1] - p / 2, top - p / 2), p, collide=False)
    # stack light on the rear left upright
    sx, sy = xs[0], ys[1]
    b.cyl(body, 0.022, 0.012, (sx, sy, top + 0.012), "black")
    b.cyl(body, 0.009, 0.10, (sx, sy, top + 0.124), "alu")
    for k, mat in enumerate(("stack_green", "stack_amber_off", "stack_red_off")):
        b.cyl(body, 0.035, 0.030, (sx, sy, top + 0.26 + k * 0.064), mat)
        b.cyl(body, 0.036, 0.002, (sx, sy, top + 0.292 + k * 0.064), "black")
    b.cyl(body, 0.035, 0.012, (sx, sy, top + 0.462), "black")
    # station sign on the front beam
    b.quad(body, 0.62, 0.097, (-0.45, ys[0] - p / 2 - 0.002, top - 0.12), G.rot_x(np.pi / 2), "sign_cell")
    # HMI screen on an arm at the left front upright, clear of the camera mast on the right
    ax, ay = xs[0], ys[0]
    b.box(body, (0.10, 0.012, 0.012), (ax + 0.12, ay, zw + 0.60), "alu_dark")
    b.box(body, (0.012, 0.012, 0.06), (ax + 0.22, ay, zw + 0.64), "alu_dark")
    rot = G.rot_z(np.radians(-12.0)) @ G.rot_x(np.radians(-8.0))
    screen_c = np.array([ax + 0.30, ay - 0.03, zw + 0.74])
    b.box(body, (0.19, 0.018, 0.115), screen_c, "monitor_black", rot=rot)
    b.quad(body, 0.36, 0.2025, screen_c + rot @ np.array([0, -0.0185, 0]), rot @ G.rot_x(np.pi / 2), "screen")
    pw, ph, pd = UR_TEACH_PENDANT
    pr = G.rot_z(np.radians(90)) @ G.rot_x(np.radians(-10))
    pc = np.array([xs[0] - p / 2 - pd / 2 - 0.005, 0.05, zw + 0.42])
    b.box(body, (pw / 2, pd / 2, ph / 2), pc, "pendant", rot=pr)
    b.box(body, (0.012, 0.030, 0.012), (xs[0] - p / 2 - 0.012, 0.05, zw + 0.42 + ph / 2 + 0.01), "steel_dark")
    b.quad(body, pw * 0.62, ph * 0.72, pc + pr @ np.array([0.03, -pd / 2 - 0.001, 0]), pr @ G.rot_x(np.pi / 2), "screen")
    return {"top": top, "legs_x": xs, "legs_y": ys, "beam_y": beam_y}


# ---------------------------------------------------------------- jig and parts

def enclosure_base(L: Layout) -> tuple[G.Mesh, G.Mesh, list]:
    """Enclosure base: walls, floor, corner bosses and two rail bosses. Returns (shell, hole walls, hole points)."""
    w, d, h = L.box_size
    wall_t, floor_t, r = 0.0028, 0.003, 0.010
    outer = G.rounded_rect(w, d, r, 6)
    inner = G.rounded_rect(w - 2 * wall_t, d - 2 * wall_t, r - wall_t, 6)
    parts = [G.ring_prism(outer, inner, floor_t, h), G.prism(outer, 0.0, floor_t)]
    darks, holes = [], []
    corner = [(sx * (w / 2 - 0.0095), sy * (d / 2 - 0.0095)) for sx in (-1, 1) for sy in (-1, 1)]
    rail = [(-0.070, 0.0), (0.070, 0.0)]
    for (x, y), top, r_out in [(c, h - 0.004, 0.0065) for c in corner] + [(c, 0.021, 0.0055) for c in rail]:
        boss = G.lathe([(0.0019, floor_t), (r_out, floor_t), (r_out, top), (0.0019, top)], 28)
        parts.append(boss.moved(pos=(x, y, 0)))
        darks.append(G.tube(0.0019, 0.0012, top - floor_t, 20).moved(pos=(x, y, floor_t)))
        darks.append(G.cylinder(0.0019, 0.0005, 20).moved(pos=(x, y, top - 0.012)))
        holes.append((x, y, top))
    return G.merge(parts), G.merge(darks), holes


def build_jig_and_parts(b: Builder, L: Layout, zw: float) -> dict:
    """The jig, and the three parts the worker handles.

    The base, the rail with its terminal blocks and the cover are separate bodies
    on free joints, so a script can carry them. Their starting pose is the
    assembled box in the jig, which is what the stills show.
    """
    body = b.spec.worldbody.add_body(name="jig")
    jx, jy = L.jig_xy
    pw, pd, pt = L.jig_plate
    b.box(body, (pw / 2, pd / 2, pt / 2), (jx, jy, zw + pt / 2), "alu", collide=True, name="jig_plate")
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.cyl(body, 0.005, 0.0015, (jx + sx * (pw / 2 - 0.015), jy + sy * (pd / 2 - 0.015), zw + pt + 0.0015), "steel_dark")
    w, d, h = L.box_size
    z0 = zw + pt
    # side stops: L-shaped hard stops on the rear and left, pins on the front
    b.box(body, (0.06, 0.008, 0.015), (jx - 0.05, jy + d / 2 + 0.010, z0 + 0.015), "steel_dark", collide=True)
    b.box(body, (0.008, 0.045, 0.015), (jx - w / 2 - 0.010, jy + 0.02, z0 + 0.015), "steel_dark", collide=True)
    for x in (-0.07, 0.07):
        b.cyl(body, 0.005, 0.012, (jx + x, jy - d / 2 - 0.008, z0 + 0.012), "steel")
    # two toggle clamps on the right side (visual only)
    for yy in (-0.045, 0.045):
        _toggle_clamp(b, body, (jx + w / 2 + 0.045, jy + yy, z0))

    cover_holes = [(sx * (w / 2 - 0.0095), sy * (d / 2 - 0.0095)) for sx in (-1, 1) for sy in (-1, 1)]
    cz = z0 + h + 0.002
    points = {"cover_holes": [(jx + x, jy + y, cz + 0.002) for x, y in cover_holes],
              "rail_holes": [(jx + x, jy, z0 + 0.0225) for x in (-0.070, 0.070)],
              "box_home": (jx, jy, z0), "cover_home": (jx, jy, cz), "rail_home": (jx, jy, z0 + 0.021)}

    # the base, on a free joint so the worker can carry it
    base_body = b.spec.worldbody.add_body(name="part_base", pos=[jx, jy, z0])
    base_body.add_freejoint(name="part_base_free")
    shell, darks, _ = enclosure_base(L)
    b.meshgeom(base_body, shell, "enclosure", (0, 0, 0))
    b.meshgeom(base_body, darks, "hole", (0, 0, 0))
    # Collision shell: floor, four walls and the screw bosses, not one solid block.
    # The rail screws are driven inside the enclosure, so a solid proxy put every pose
    # that reaches them in collision and made the reach numbers meaningless.
    wall_t, floor_t = 0.0028, 0.003
    wall_h = (h - floor_t) / 2
    shells = [((w / 2, d / 2, floor_t / 2), (0.0, 0.0, floor_t / 2), "box_collision")]
    shells += [((w / 2, wall_t / 2, wall_h), (0.0, sy * (d / 2 - wall_t / 2), floor_t + wall_h), None)
               for sy in (-1, 1)]
    shells += [((wall_t / 2, d / 2 - wall_t, wall_h), (sx * (w / 2 - wall_t / 2), 0.0, floor_t + wall_h), None)
               for sx in (-1, 1)]
    for half, at, name in shells:
        g = b.box(base_body, half, at, collide=True, group=GROUP_COLLISION, name=name)
        g.density = 2700.0
    for (bx, by), top, r_out in ([((sx * (w / 2 - 0.0095), sy * (d / 2 - 0.0095)), h - 0.004, 0.0065)
                                  for sx in (-1, 1) for sy in (-1, 1)]
                                 + [((x, 0.0), 0.021, 0.0055) for x in (-0.070, 0.070)]):
        g = b.cyl(base_body, r_out, (top - floor_t) / 2, (bx, by, floor_t + (top - floor_t) / 2),
                  collide=True, group=GROUP_COLLISION)
        g.density = 2700.0

    # the rail with its four terminal blocks, pre-fitted by the worker
    rail_body = b.spec.worldbody.add_body(name="part_rail", pos=[jx, jy, z0 + 0.021])
    rail_body.add_freejoint(name="part_rail_free")
    _din_rail(b, rail_body, (0.0, 0.0, 0.0), 0.160, with_blocks=True)
    # the rail strip and the block cluster separately, so the two screw slots at
    # x = +-70 mm stay open: one box over the whole rail hid them.
    g = b.box(rail_body, (0.080, 0.0175, 0.00375), (0, 0, 0.00375), collide=True, group=GROUP_COLLISION)
    g.density = 2000.0
    g = b.box(rail_body, (0.0104, 0.02125, 0.022), (0, 0, 0.0075 + 0.022), collide=True, group=GROUP_COLLISION)
    g.density = 1200.0

    if not L.with_cover:
        return points

    cover_body = b.spec.worldbody.add_body(name="part_cover", pos=[jx, jy, cz])
    cover_body.add_freejoint(name="part_cover_free")
    plate, walls = G.plate_with_holes(w, d, 0.004, [(x, y, G.circle(0.0023, 24)) for x, y in cover_holes], corner_r=0.010)
    lip = G.ring_prism(G.rounded_rect(w - 0.0062, d - 0.0062, 0.007, 6), G.rounded_rect(w - 0.0102, d - 0.0102, 0.005, 6), -0.006, 0.0)
    b.meshgeom(cover_body, G.merge([plate, lip.moved(pos=(0, 0, -0.002))]), "enclosure", (0, 0, 0), name="cover")
    b.meshgeom(cover_body, walls, "hole", (0, 0, 0))
    g = b.box(cover_body, (w / 2, d / 2, 0.004), (0, 0, 0), collide=True, group=GROUP_COLLISION)
    g.density = 300.0
    # screws already driven, if the scene is posed mid work
    for x, y in (cover_holes[1], cover_holes[2])[:L.driven_screws]:
        _screw_head(b, cover_body, (x, y, 0.002))
    return points


def _toggle_clamp(b: Builder, body, base) -> None:
    x, y, z = base
    b.box(body, (0.022, 0.016, 0.003), (x, y, z + 0.003), "zinc")
    for side in (-1, 1):
        b.box(body, (0.012, 0.002, 0.018), (x, y + side * 0.009, z + 0.021), "zinc")
    b.box(body, (0.030, 0.006, 0.004), (x - 0.024, y, z + 0.040), "zinc", rot=G.rot_y(np.radians(-8)))
    b.cyl(body, 0.0035, 0.012, (x - 0.052, y, z + 0.030), "zinc")
    b.cyl(body, 0.006, 0.004, (x - 0.052, y, z + 0.016), "rubber")
    b.box(body, (0.004, 0.004, 0.028), (x + 0.012, y, z + 0.060), "zinc", rot=G.rot_y(np.radians(-25)))
    b.capsule(body, 0.0065, (x + 0.020, y, z + 0.075), (x + 0.034, y, z + 0.104), "grip_red")


def _din_rail(b: Builder, body, pos, length: float, with_blocks: bool) -> None:
    """35 x 7.5 mm top-hat rail with two mounting slots and optional terminal blocks (scene choice sizes)."""
    x, y, z = pos
    plate, walls = G.plate_with_holes(length, 0.027, 0.001, [(-0.070, 0.0, G.stadium(0.012, 0.0053, 8)),
                                                               (0.070, 0.0, G.stadium(0.012, 0.0053, 8))])
    b.meshgeom(body, plate, "zinc", (x, y, z + 0.0005))
    b.meshgeom(body, walls, "hole", (x, y, z + 0.0005))
    for side in (-1, 1):
        b.box(body, (length / 2, 0.0005, 0.00375), (x, y + side * 0.0135, z + 0.00375), "zinc")
        b.box(body, (length / 2, 0.002, 0.0005), (x, y + side * 0.0155, z + 0.0070), "zinc")
    if not with_blocks:
        return
    prof = np.array([(0, 0), (0.0425, 0), (0.0425, 0.028), (0.036, 0.028), (0.036, 0.044), (0.026, 0.044),
                     (0.026, 0.034), (0.0165, 0.034), (0.0165, 0.044), (0.0065, 0.044), (0.0065, 0.028), (0, 0.028)])
    block = G.extrude_x(prof - [0.02125, 0], -0.0026, 0.0026)
    for i in range(4):
        bx = x - 0.0078 + i * 0.0052
        mat = "terminal_pe" if i == 3 else "terminal_grey"
        b.meshgeom(body, block, mat, (bx, y, z + 0.0075))
        for yy in (-0.0105, 0.0105):
            b.cyl(body, 0.0019, 0.001, (bx, y + yy, z + 0.0075 + 0.028 + 0.002), "zinc")


def _screw_head(b: Builder, body, pos, material: str = "zinc") -> None:
    x, y, z = pos
    head = G.lathe([(0.0, 0.0), (0.0038, 0.0), (0.0038, 0.0012), (0.0030, 0.0028), (0.0, 0.0031)], 24)
    b.meshgeom(body, head, material, (x, y, z))
    b.box(body, (0.0022, 0.0005, 0.0003), (x, y, z + 0.0030), "hole")
    b.box(body, (0.0005, 0.0022, 0.0003), (x, y, z + 0.0030), "hole")


def screw_mesh() -> G.Mesh:
    """Pan head screw, about M4 x 12, head at z = 0 pointing up, tip at z = -0.012 (scene choice size)."""
    return G.lathe([(0.0, -0.0125), (0.0014, -0.0115), (0.0020, -0.0105), (0.0020, 0.0), (0.0038, 0.0),
                    (0.0038, 0.0012), (0.0030, 0.0028), (0.0, 0.0031)], 24)


def build_feeder_tray_prep(b: Builder, L: Layout, zw: float) -> dict:
    body = b.spec.worldbody.add_body(name="props")
    # screw presenter on the robot side (scene choice sizes)
    fx, fy = L.feeder_xy
    fw, fd, fh = 0.15, 0.24, 0.16
    b.box(body, (fw / 2, fd / 2, fh / 2), (fx, fy, zw + fh / 2), "light_plastic", collide=True, name="feeder")
    b.box(body, (fw / 2 + 0.001, fd / 2 + 0.001, 0.012), (fx, fy, zw + 0.012), "grey_plastic")
    b.box(body, (0.045, 0.06, 0.035), (fx + 0.02, fy + 0.04, zw + fh + 0.035), "grey_plastic")
    b.box(body, (0.040, 0.055, 0.001), (fx + 0.02, fy + 0.04, zw + fh + 0.069), "hole")
    b.box(body, (0.006, 0.07, 0.004), (fx - 0.04, fy - 0.02, zw + fh + 0.004), "steel")
    pick = np.array([fx - 0.04, fy - 0.095, zw + fh + 0.010])
    b.box(body, (0.012, 0.012, 0.010), pick - [0, 0, 0.002], "steel_dark")
    b.meshgeom(body, screw_mesh(), "zinc", pick + [0, 0, 0.0085])
    b.sphere(body, 0.004, (fx + 0.06, fy - fd / 2 - 0.001, zw + 0.12), "stack_green")
    # out tray with two finished boxes
    tx, ty = L.tray_xy
    tray = G.merge([G.ring_prism(G.rounded_rect(0.30, 0.24, 0.02), G.rounded_rect(0.29, 0.23, 0.015), 0.004, 0.07),
                    G.prism(G.rounded_rect(0.30, 0.24, 0.02), 0.0, 0.004)])
    b.meshgeom(body, tray, "tote_grey", (tx, ty, zw))
    w, d, h = L.box_size
    done_shell, _, _ = enclosure_base(L)
    for i, (ox, oy, yaw) in enumerate(((-0.02, 0.0, 0.0), (0.03, 0.01, 0.08))):
        rot = G.rot_z(yaw)
        c = np.array([tx + ox, ty + oy, zw + 0.004 + i * (h + 0.006)])
        b.meshgeom(body, done_shell, "enclosure", c, rot)
        plate, walls = G.plate_with_holes(w, d, 0.004, [], corner_r=0.010)
        b.meshgeom(body, plate, "enclosure", c + [0, 0, h + 0.002], rot)
        for sx in (-1, 1):
            for sy in (-1, 1):
                _screw_head(b, body, c + rot @ np.array([sx * (w / 2 - 0.0095), sy * (d / 2 - 0.0095), h + 0.004]))
    # prep area: small ESD mat with a rail being fitted and loose blocks
    px, py = L.prep_xy
    b.box(body, (0.16, 0.12, 0.0015), (px, py, zw + 0.0015), "esd")
    _din_rail(b, body, (px, py, zw + 0.003), 0.160, with_blocks=True)
    return {"feeder_pick": pick + [0, 0, 0.0085]}


def build_parts_rack(b: Builder, L: Layout, zw: float) -> dict:
    """Floor-standing flow rack on the worker's left: two tilted shelves, four totes.

    Returns the pick point of each tote, which is where a cycle starts the parts.
    """
    body = b.spec.worldbody.add_body(name="rack")
    picks: dict = {}
    # 30 cm further toward the worker's side than first placed (y0 was -0.52): against
    # the bench's left end, the far totes were reachable only from inside the bench's
    # footprint, where the script had him standing since gate 2 (scene choice)
    x_in, x_out, y0, y1 = -0.86, -1.26, -0.82, -0.18
    p = 0.03
    for x in (x_in, x_out):
        for y in (y0, y1):
            h = zw + (0.42 if x == x_out else 0.30)
            profile_bar(b, body, (x, y, 0.05), (x, y, h), p, collide=False)
            b.cyl(body, 0.03, 0.02, (x, y, 0.03), "rubber")
    tilt = np.radians(14.0)
    for lvl, zc in enumerate((zw - 0.16, zw + 0.16)):
        rot = G.rot_y(tilt)
        c = np.array([(x_in + x_out) / 2, (y0 + y1) / 2, zc])
        b.box(body, (0.22, (y1 - y0) / 2, 0.006), c, "alu_dark", rot=rot)
        for k, (mat, label) in enumerate(((("tote_blue", "base"), ("tote_blue", "cover")),
                                          (("tote_red", "rail"), ("tote_yellow", "tb")))[lvl]):
            ty = y0 + 0.16 + k * 0.32
            tc = np.array([c[0], ty, zc]) + rot @ np.array([0.0, 0.0, 0.006])
            tote = G.merge([G.ring_prism(G.rounded_rect(0.30, 0.24, 0.02), G.rounded_rect(0.29, 0.23, 0.015), 0.004, 0.11),
                            G.prism(G.rounded_rect(0.30, 0.24, 0.02), 0.0, 0.004)])
            b.meshgeom(body, tote, mat, tc, rot)
            b.quad(body, 0.10, 0.04, tc + rot @ np.array([0.1505, 0.0, 0.075]), rot @ G.rot_z(np.pi / 2) @ G.rot_x(np.pi / 2), f"label_{label}")
            _tote_contents(b, body, label, tc, rot, L)
            picks[f"pick_{label}"] = tuple(tc + rot @ np.array([0.0, 0.0, 0.055]))
    return picks


def _tote_contents(b: Builder, body, label: str, c, rot, L: Layout) -> None:
    w, d, h = L.box_size
    if label == "base":
        shell, _, _ = enclosure_base(L)
        for i in range(2):
            b.meshgeom(body, shell, "enclosure", c + rot @ np.array([-0.03, 0.0, 0.006 + i * 0.012]),
                       rot @ G.rot_z(np.pi / 2 + 0.03 * i))
    elif label == "cover":
        plate, walls = G.plate_with_holes(w, d, 0.004, [(sx * (w / 2 - 0.0095), sy * (d / 2 - 0.0095), G.circle(0.0023, 16))
                                                      for sx in (-1, 1) for sy in (-1, 1)], corner_r=0.010)
        for i in range(6):
            b.meshgeom(body, plate, "enclosure", c + rot @ np.array([0.0, 0.0, 0.012 + i * 0.0045]), rot @ G.rot_z(np.pi / 2 + 0.02 * (i % 2)))
    elif label == "rail":
        for i in range(5):
            b.box(body, (0.08, 0.0175, 0.00375), c + rot @ np.array([-0.05 + i * 0.001, -0.08 + i * 0.04, 0.012]), "zinc",
                  rot=rot @ G.rot_z(np.pi / 2 + 0.05 * (i - 2)))
    else:
        rng = np.random.default_rng(4)
        for i in range(26):
            off = np.array([rng.uniform(-0.13, 0.13), rng.uniform(-0.10, 0.10), 0.02 + rng.uniform(0, 0.03)])
            b.box(body, (0.0026, 0.021, 0.014), c + rot @ off, "terminal_pe" if i % 7 == 0 else "terminal_grey",
                  rot=rot @ G.rot_z(rng.uniform(0, np.pi)) @ G.rot_x(rng.uniform(-0.4, 0.4)))


# ---------------------------------------------------------------- robot, tool, cameras

def screwdriver_meshes() -> dict:
    """OnRobot Screwdriver shape in the tool frame: z out of the robot flange, bit along -y."""
    ax = SCREW_AXIS_OFFSET
    zc = QUICK_CHANGER_T + SD_DEPTH / 2
    y_bot = -(SD_FLANGE_ABOVE_TIP - SD_HOLDER - SD_NOSE_LEN)
    y_top = y_bot + SD_BODY_HEIGHT
    n = 36

    def ring(y, scale=1.0, lift=0.0):
        loop = G.superellipse(SD_DEPTH / 2 * scale, SD_WIDTH / 2 * scale, 3.2, n)  # (u = z, v = x)
        return np.column_stack([loop[:, 1], np.full(n, y + lift), zc + loop[:, 0]])

    lower = G.loft([ring(y_bot, 0.94), ring(y_bot + 0.004), ring(y_bot + 0.018), ring(y_bot + 0.0185, 0.985),
                    ring(y_bot + 0.021, 0.985), ring(y_bot + 0.0215)])
    body = G.loft([ring(y_bot + 0.020), ring(y_top - 0.070)])
    cap_secs = []
    for dy, s in ((0.072, 1.0), (0.05, 0.995), (0.03, 0.97), (0.016, 0.90), (0.007, 0.74), (0.0015, 0.42)):
        sec = ring(y_top - dy, s)
        sec[:, 1] += (sec[:, 2] - zc) * 0.08  # top slopes up toward the flange side, as in the drawing
        cap_secs.append(sec)
    cap = G.loft(cap_secs)
    nose = G.lathe([(0.0, 0.0), (SD_NOSE_D / 2 * 0.72, 0.0), (SD_NOSE_D / 2 * 0.86, 0.004),
                    (SD_NOSE_D / 2, 0.016), (SD_NOSE_D / 2, SD_NOSE_LEN), (0.0, SD_NOSE_LEN)], 32)
    holder = G.lathe([(0.0, 0.0), (SD_HOLDER / 2 * 0.8, 0.0), (SD_HOLDER / 2, 0.002),
                      (SD_HOLDER / 2, SD_HOLDER), (0.0, SD_HOLDER)], 24)
    y_tip = -SD_FLANGE_ABOVE_TIP
    return {"lower": lower, "body": body, "cap": cap, "nose_raw": nose, "holder_raw": holder,
            "axis_z": ax, "y_bottom": y_bot, "y_tip": y_tip}


def build_robot(b: Builder, L: Layout, zw: float) -> dict:
    rx, ry = L.robot_xy
    pw, pd, pt = L.robot_plate
    mount = b.spec.worldbody.add_body(name="robot_mount")
    b.box(mount, (pw / 2, pd / 2, pt / 2), (rx, ry, zw + pt / 2), "steel_dark", collide=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.cyl(mount, 0.006, 0.002, (rx + sx * (pw / 2 - 0.018), ry + sy * (pd / 2 - 0.018), zw + pt + 0.002), "zinc")
    ur = mujoco.MjSpec.from_file(str(MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"))
    for key in list(ur.keys):
        ur.delete(key)
    frame = b.spec.worldbody.add_frame(pos=[rx, ry, zw + pt], quat=G.mat_to_quat(G.rot_z(-np.pi / 2)))
    b.spec.attach(ur, prefix=ROBOT_PREFIX, frame=frame)
    w3 = b.spec.body(ROBOT_PREFIX + "wrist_3_link")
    # attachment_site in wrist_3_link: pos (0, 0.1, 0), quat (-1, 1, 0, 0) (Menagerie ur5e.xml)
    site_R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    site_p = np.array([0.0, 0.1, 0.0])

    def to_w3(p, R=np.eye(3)):
        return site_p + site_R @ np.asarray(p, float), site_R @ R

    sd = screwdriver_meshes()
    grp = GROUP_ROBOT
    for key, mat in (("lower", "tool_light"), ("body", "tool_dark"), ("cap", "tool_light")):
        p, R = to_w3((0, 0, 0))
        b.meshgeom(w3, sd[key], mat, p, R, group=grp)
    # quick changer and the mounting boss
    p, R = to_w3((0, 0, QUICK_CHANGER_T / 2))
    b.cyl(w3, 0.037, QUICK_CHANGER_T / 2, p, "steel_dark", rot=R, group=grp)
    p, R = to_w3((0, 0.0, QUICK_CHANGER_T + 0.004))
    b.box(w3, (0.030, 0.034, 0.005), p, "tool_dark", rot=R, group=grp)
    # nose and bit holder along -y of the tool frame, centred on the screw axis
    lathe_to_tool = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # lathe +z maps to tool -y
    y_bot = sd["y_bottom"]
    p, R = to_w3((0, y_bot + 0.001, SCREW_AXIS_OFFSET), lathe_to_tool)
    b.meshgeom(w3, sd["nose_raw"], "tool_light", p, R, group=grp)
    p, R = to_w3((0, y_bot - SD_NOSE_LEN + 0.0005, SCREW_AXIS_OFFSET), lathe_to_tool)
    b.meshgeom(w3, sd["holder_raw"], "steel", p, R, group=grp)
    # screw held on the bit, head against the holder
    head_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # screw mesh +z (head up) maps to tool +y
    p, R = to_w3((0, -SD_FLANGE_ABOVE_TIP - 0.0031, SCREW_AXIS_OFFSET), head_up)
    b.meshgeom(w3, screw_mesh(), "zinc", p, R, group=grp, name="screw_on_bit")
    # tool tip site: z along the bit
    p, R = to_w3((0, -SD_FLANGE_ABOVE_TIP, SCREW_AXIS_OFFSET), lathe_to_tool)
    w3.add_site(name="tool_tip", pos=p, quat=G.mat_to_quat(R), size=[0.003, 0, 0], group=4)
    # collision stand-ins for the tool
    cp, cR = to_w3((0, (y_bot + y_bot + SD_BODY_HEIGHT) / 2, QUICK_CHANGER_T + SD_DEPTH / 2), lathe_to_tool)
    g = b.cyl(w3, 0.060, SD_BODY_HEIGHT / 2, cp, rot=cR, group=GROUP_COLLISION)
    g.contype, g.conaffinity = 1, 1
    # The nose, the bit holder and the screw on it. R1 covers the arm, the screwdriver,
    # the wrist camera and the screw (docs/design.md section 4), and without these the bit tip
    # sits 74 mm outside every collision geom, so the one contact most likely to happen,
    # a poke with the screw, would be scored as clear.
    for y0, y1, radius in ((y_bot, y_bot - SD_NOSE_LEN, SD_NOSE_D / 2),
                           (y_bot - SD_NOSE_LEN, -SD_FLANGE_ABOVE_TIP, 0.0085)):
        p0 = to_w3((0.0, y0, SCREW_AXIS_OFFSET))[0]
        p1 = to_w3((0.0, y1, SCREW_AXIS_OFFSET))[0]
        g = b.capsule(w3, radius, p0, p1, group=GROUP_COLLISION)
        g.contype, g.conaffinity = 1, 1
    # wrist camera (D405) on a bracket at the +x side, 75 mm from the screw axis
    cam_tool = np.array([0.078, -SD_FLANGE_ABOVE_TIP + 0.052, SCREW_AXIS_OFFSET])
    look = np.array([0.0, -SD_FLANGE_ABOVE_TIP - 0.035, SCREW_AXIS_OFFSET])
    fwd = look - cam_tool
    fwd /= np.linalg.norm(fwd)
    up_hint = np.array([0.0, 1.0, 0.0])  # tool body toward the top of the image
    xc = np.cross(fwd, up_hint)
    xc /= np.linalg.norm(xc)
    yc = np.cross(xc, fwd)
    cam_R = np.column_stack([xc, yc, -fwd])
    hx, hy, hz = D405_SIZE[0] / 2, D405_SIZE[1] / 2, D405_SIZE[2] / 2
    p, R = to_w3(cam_tool, cam_R)
    b.box(w3, (hx, hy, hz), p, "black", rot=R, group=grp)
    g = b.box(w3, (hx, hy, hz), p, rot=R, group=GROUP_COLLISION)  # R1 counts the camera too
    g.contype, g.conaffinity = 1, 1
    for off in (-0.009, 0.009):
        lp, lR = to_w3(cam_tool + cam_R @ np.array([off, 0.0, hz + 0.0004]), cam_R)
        b.cyl(w3, 0.0045, 0.0005, lp, "glass", rot=lR, group=grp)
    for bpos, bhalf in (((SD_WIDTH / 2 + 0.006, y_bot + 0.004, SCREW_AXIS_OFFSET), (0.006, 0.020, 0.016)),
                        ((0.060, y_bot - 0.010, SCREW_AXIS_OFFSET), (0.016, 0.004, 0.012))):
        bp, bR = to_w3(np.array(bpos))
        b.box(w3, bhalf, bp, "alu", rot=bR, group=grp)
    cam = w3.add_camera(name="wrist_cam", pos=p, quat=G.mat_to_quat(R), fovy=D405_FOVY)
    del cam
    return {"tool_offset_axis": SCREW_AXIS_OFFSET}


EYES_VARIANTS = {
    # name: (camera position, aim point offset above the worktop, mount style)
    # Both are scene choices. A front camera has to see over or past the worker's head,
    # which sets how low the angle can go; gate 1 measures coverage for the one we keep.
    "centre": ((0.0, -1.22, 1.80), (0.0, -0.06, 0.04), "centre"),
    "offset": ((0.55, -1.15, 1.45), (0.0, -0.06, 0.04), "offset"),
    "top": ((0.0, -0.36, 1.02), (0.0, -0.08, 0.09), "beam"),
    "corner": ((1.06, -0.95, 1.54), (0.0, -0.06, 0.04), "corner"),
    "corner_high": ((0.95, -0.62, 1.58), (0.0, -0.06, 0.04), "corner"),
    "pole": ((1.15, -0.90, 2.13), (0.0, -0.06, 0.04), "pole"),
    # Chosen 2026-09-18 by scripts/eyes_optimise.py: of 76 pole positions this one sees
    # the most of the space the arm and a person can share (86.6 %, the old pole 84.5 %)
    # while keeping the space behind the robot in the picture (82.0 %, was 79.4 %).
    "pole_far": ((1.70, -1.50, 2.48), (0.0, -0.06, 0.04), "pole"),
}


def build_eyes_camera(b: Builder, L: Layout, zw: float, frame_info: dict) -> dict:
    """Overhead RealSense D455 on a mast and boom in front of the bench."""
    body = b.spec.worldbody.add_body(name="eyes")
    y_front = frame_info["legs_y"][0]
    x_right = frame_info["legs_x"][1]
    top = frame_info["top"]
    p = L.profile
    rel_pos, rel_target, style = EYES_VARIANTS[L.eyes_variant]
    pos = np.array([rel_pos[0], rel_pos[1], zw + rel_pos[2]])
    target = np.array([rel_target[0], rel_target[1], zw + rel_target[2]])
    if style == "centre":
        mast_top = pos[2] + 0.06
        profile_bar(b, body, (0.0, y_front, top), (0.0, y_front, mast_top), p, collide=False)
        profile_bar(b, body, (0.0, y_front - p / 2, mast_top - p / 2), (pos[0], pos[1], mast_top - p / 2), p, collide=False)
        profile_bar(b, body, (0.0, y_front + 0.01, mast_top - 0.42), (pos[0], pos[1] + 0.30, mast_top - p), p, collide=False)
        b.box(body, (0.035, 0.035, 0.004), (pos[0], pos[1], mast_top - p - 0.004), "alu_dark")
        b.box(body, (0.006, 0.016, 0.026), (pos[0], pos[1], mast_top - p - 0.030), "alu_dark")
    elif style == "offset":
        mast_top = pos[2] + 0.06
        profile_bar(b, body, (x_right, y_front, top), (x_right, y_front, mast_top), p, collide=False)
        profile_bar(b, body, (x_right, y_front - p / 2, mast_top - p / 2), (pos[0], pos[1], mast_top - p / 2), p, collide=False)
        profile_bar(b, body, (x_right, y_front, mast_top - 0.40), (pos[0] + 0.10, pos[1] + 0.26, mast_top - p), p, collide=False)
        b.box(body, (0.035, 0.035, 0.004), (pos[0], pos[1], mast_top - p - 0.004), "alu_dark")
        b.box(body, (0.006, 0.016, 0.026), (pos[0], pos[1], mast_top - p - 0.030), "alu_dark")
    elif style == "pole":
        # floor pole beside the bench, standing behind the camera so it stays out of the view
        back = np.array([pos[0] - target[0], pos[1] - target[1], 0.0])
        back /= np.linalg.norm(back)
        foot = pos[:2] + back[:2] * 0.085
        profile_bar(b, body, (foot[0], foot[1], 0.02), (foot[0], foot[1], pos[2] + 0.16), 0.060, collide=True)
        b.box(body, (0.13, 0.13, 0.008), (foot[0], foot[1], 0.008), "steel_dark", collide=True)
        for sx in (-1, 1):
            for sy in (-1, 1):
                b.cyl(body, 0.008, 0.004, (foot[0] + sx * 0.095, foot[1] + sy * 0.095, 0.018), "zinc")
        arm_mid = pos + back * 0.045 + np.array([0.0, 0.0, 0.012])
        b.box(body, (0.050, 0.014, 0.010), arm_mid, "alu_dark",
              rot=G.rot_z(float(np.arctan2(back[1], back[0]))))
    elif style == "corner":
        mast_top = pos[2] + 0.06
        profile_bar(b, body, (x_right, y_front, top), (x_right, y_front, mast_top), p, collide=False)
        profile_bar(b, body, (x_right, y_front, mast_top - p / 2), (pos[0], pos[1], mast_top - p / 2), p, collide=False)
        profile_bar(b, body, (x_right, y_front, mast_top - 0.38), (pos[0] - 0.06, pos[1] + 0.18, mast_top - p), p, collide=False)
        b.box(body, (0.035, 0.035, 0.004), (pos[0], pos[1], mast_top - p - 0.004), "alu_dark")
        b.box(body, (0.006, 0.016, 0.026), (pos[0], pos[1], mast_top - p - 0.030), "alu_dark")
    else:
        b.box(body, (0.015, 0.03, 0.008), (pos[0], y_front + 0.02, pos[2] + 0.05), "alu_dark")
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    xc = np.cross(fwd, [0, 0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(xc, fwd)
    R = np.column_stack([xc, yc, -fwd])
    sw, sh, sd = D455_SIZE
    b.box(body, (sw / 2, sh / 2, sd / 2), pos - fwd * sd / 2, "black", rot=R)
    for off in (-0.0475, 0.0475, -0.02):
        b.cyl(body, 0.006, 0.0006, pos + R @ [off, 0, 0] + fwd * 0.0004, "glass", rot=R)
    body.add_camera(name="eyes_cam", pos=pos + fwd * 0.002, quat=G.mat_to_quat(R), fovy=D455_FOVY)
    pitch = float(np.degrees(np.arcsin(-fwd[2])))
    return {"eyes_pos": pos, "eyes_target": target, "eyes_pitch_deg": pitch}


def add_lights(b: Builder, zw: float, top: float) -> None:
    wb = b.spec.worldbody
    L_ = mujoco.mjtLightType
    for name, x, shadow in (("key_l", -0.52, True), ("key_r", 0.52, True)):
        wb.add_light(name=name, type=L_.mjLIGHT_SPOT, pos=[x, 0.0, top - 0.06], dir=[-0.35 * np.sign(x), -0.05, -1.0],
                     diffuse=[0.55, 0.52, 0.47], specular=[0.15, 0.15, 0.15], ambient=[0, 0, 0],
                     cutoff=48.0, exponent=2.0, castshadow=shadow, bulbradius=0.25, attenuation=[1.0, 0.0, 0.0])
    wb.add_light(name="hall", type=L_.mjLIGHT_DIRECTIONAL, pos=[0, 0, 8], dir=[0.35, -0.30, -1.0],
                 diffuse=[0.42, 0.44, 0.47], specular=[0.05, 0.05, 0.05], ambient=[0, 0, 0], castshadow=False)
    wb.add_light(name="fill_front", type=L_.mjLIGHT_SPOT, pos=[-1.8, -3.2, 3.2], dir=[0.45, 0.8, -0.55],
                 diffuse=[0.30, 0.29, 0.28], specular=[0.05, 0.05, 0.05], cutoff=40.0, exponent=1.0, castshadow=False)
    wb.add_light(name="rim", type=L_.mjLIGHT_SPOT, pos=[2.4, 2.6, 2.9], dir=[-0.6, -0.65, -0.45],
                 diffuse=[0.28, 0.30, 0.34], specular=[0.12, 0.12, 0.12], cutoff=35.0, exponent=1.0, castshadow=False)
    for x in (-3.6, 3.6):
        wb.add_light(name=f"station_{'l' if x < 0 else 'r'}", type=L_.mjLIGHT_SPOT, pos=[x, 0.3, 2.9],
                     dir=[0, 0, -1], diffuse=[0.55, 0.52, 0.48], specular=[0.05, 0.05, 0.05], cutoff=60.0,
                     exponent=1.5, castshadow=False)
    wb.add_light(name="windows", type=L_.mjLIGHT_DIRECTIONAL, pos=[0, 6, 5], dir=[0.0, -1.0, -0.35],
                 diffuse=[0.16, 0.17, 0.19], specular=[0, 0, 0], castshadow=False)


def add_film_cameras(b: Builder, zw: float) -> None:
    wb = b.spec.worldbody

    def cam(name, pos, target, fovy, up=(0.0, 0.0, 1.0)):
        x, y = np.split(np.array(G.look_at_xyaxes(pos, target, up)), 2)
        R = np.column_stack([x, y, np.cross(x, y)])
        wb.add_camera(name=name, pos=list(pos), quat=G.mat_to_quat(R), fovy=fovy)

    cam("hero", (-1.16, -1.34, 1.80), (0.07, -0.05, zw + 0.02), 38.0)
    cam("side", (2.35, -0.35, 1.62), (0.0, -0.02, zw + 0.20), 38.0)
    cam("top", (0.0, -0.20, 6.2), (0.0, -0.18, zw), 24.0, up=(0.0, 1.0, 0.0))
    cam("closeup", (-0.42, 0.10, zw + 0.30), (0.02, -0.12, zw + 0.10), 40.0)
    cam("wide", (-3.9, -5.6, 2.7), (0.1, 0.4, 1.0), 46.0)
    # gate 2 onward: wide enough to hold the rack, the worker and the bench in one shot,
    # so a side step to the rack stays in frame. The gate 1 hero is too tight for that.
    cam("cycle_hero", (-2.52, -2.42, 2.12), (-0.25, -0.24, 1.20), 39.0)


def set_visual(spec: mujoco.MjSpec) -> None:
    v = spec.visual
    v.global_.offwidth = 1920
    v.global_.offheight = 1080
    v.quality.shadowsize = 8192
    v.quality.offsamples = 8
    v.quality.numslices = 48
    v.quality.numstacks = 32
    v.headlight.ambient = [0.16, 0.16, 0.17]
    v.headlight.diffuse = [0.10, 0.10, 0.10]
    v.headlight.specular = [0.0, 0.0, 0.0]
    spec.stat.extent = 8.0
    spec.stat.center = [0.0, 0.0, 1.0]
    v.map.znear = 0.0015
    v.map.zfar = 12.0
    v.map.fogstart = 1.4
    v.map.fogend = 5.5
    v.rgba.fog = [0.56, 0.58, 0.60, 1.0]
    v.rgba.haze = [0.40, 0.42, 0.44, 1.0]


WORKPIECE_GROUP = GROUP_WORKER   # renders with the robot, never counts as a robot link


def robot_only_model(layout: Layout | None = None, with_workpiece: bool = True) -> mujoco.MjModel:
    """The robot, its tool and its cameras on their mount, and nothing else.

    This is what the planner is allowed to hold: its own robot model (docs/design.md
    section 8.1). It carries no worker and no station, so nothing in it can leak
    the person's geometry. The worktop height comes from the same layout, so the
    mount sits where it does in the cell.

    With the workpiece, it also carries the three parts as plain shapes on mocap
    bodies, from the product drawing: the planner puts them where it last fitted the
    box, renders them with its own arm, and calls only what is in front of that the
    person. They are outside the collision group, so no motion code mistakes them
    for a link.
    """
    L = layout or Layout()
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    set_visual(spec)
    b = Builder(spec)
    add_materials(b)
    zw = _worktop_for(L)
    build_robot(b, L, zw)
    if with_workpiece:
        _workpiece_shapes(b, L)
    return spec.compile()


def _workpiece_shapes(b: Builder, L: Layout) -> None:
    """The base, the rail and the cover as boxes, each on its own mocap body at its
    part origin, parked out of sight until the planner places them."""
    w, d, h = L.box_size
    wall, floor = 0.0028, 0.003
    parts = {
        "wp_base": [((w / 2, d / 2, floor / 2), (0.0, 0.0, floor / 2))]
        + [((w / 2, wall / 2, (h - floor) / 2), (0.0, s * (d / 2 - wall / 2), floor + (h - floor) / 2)) for s in (-1, 1)]
        + [((wall / 2, d / 2 - wall, (h - floor) / 2), (s * (w / 2 - wall / 2), 0.0, floor + (h - floor) / 2)) for s in (-1, 1)],
        "wp_rail": [((0.080, 0.0175, 0.00375), (0.0, 0.0, 0.00375)),
                    ((0.0104, 0.02125, 0.022), (0.0, 0.0, 0.0075 + 0.022))],
        "wp_cover": [((w / 2, d / 2, 0.004), (0.0, 0.0, 0.0))],
    }
    for name, boxes in parts.items():
        body = b.spec.worldbody.add_body(name=name, pos=[0.0, 0.0, -10.0], mocap=True)
        for half, at in boxes:
            g = b.box(body, half, at, "enclosure", group=WORKPIECE_GROUP)
            g.contype, g.conaffinity = 0, 0


def _worktop_for(L: Layout) -> float:
    """Worktop height for a layout, by measuring the worker the same way build() does."""
    spec = mujoco.MjSpec()
    k, _ = attach_worker(spec, pos=[0.0, 0.0, 1.0], yaw=0.0)
    from .pose import measure_elbow_height
    return measure_elbow_height(k) - L.elbow_to_worktop


def build(layout: Layout | None = None, assets_dir=None) -> Scene:
    L = layout or Layout()
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    set_visual(spec)
    b = Builder(spec, assets_dir)
    add_materials(b)
    # the worker first: the bench height follows its measured elbow height
    k, upright = attach_worker(spec, pos=[L.worker_xy[0], L.worker_xy[1], 1.0], yaw=np.pi / 2)
    from .pose import measure_elbow_height
    elbow = measure_elbow_height(k)
    zw = elbow - L.elbow_to_worktop
    build_hall(b)
    frame_info = build_bench(b, L, zw)
    points = build_jig_and_parts(b, L, zw)
    points.update(build_feeder_tray_prep(b, L, zw))
    points.update(build_parts_rack(b, L, zw))
    build_robot(b, L, zw)
    points.update(build_eyes_camera(b, L, zw, frame_info))
    add_lights(b, zw, frame_info["top"])
    add_film_cameras(b, zw)
    model = spec.compile()
    points["elbow_height"] = elbow
    return Scene(spec, model, L, zw, k, points)


def write_obj(path, v, f, uv=None) -> Path:
    """Write a triangle mesh as an OBJ file with vertex normals, area weighted, as
    MuJoCo computes them itself: Gazebo's mesh loader wants them in the file."""
    path = Path(path)
    v, fc = np.asarray(v, float).reshape(-1, 3), np.asarray(f, int).reshape(-1, 3)
    fn = np.cross(v[fc[:, 1]] - v[fc[:, 0]], v[fc[:, 2]] - v[fc[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, fc[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    with open(path, "w") as out:
        out.write(f"o {path.stem}\n")
        out.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
        out.writelines(f"vn {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in vn)
        if uv is not None:
            out.writelines(f"vt {a:.6f} {b:.6f}\n" for a, b in np.asarray(uv, float).reshape(-1, 2))
            out.writelines(f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}\n" for a, b, c in fc)
        else:
            out.writelines(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}\n" for a, b, c in fc)
    return path.resolve()


def export_mjcf(out_dir, layout: Layout | None = None) -> Path:
    """Save the whole cell as one MJCF file with its textures and meshes beside it, so
    other tools can load it: Gazebo's mjcf2sdf converter and mujoco_ros2_control.
    Returns the path of the XML. The model it describes is the one build() makes."""
    import re
    import shutil
    out = Path(out_dir)
    assets = out / "assets"
    scene = build(layout, assets_dir=assets)
    # geom groups only choose what a viewer shows; the converter takes group 0 as
    # "visual" and group 3 as "collision" and drops every other group, so the
    # worker's group 1 and the robot's group 2 skins are folded into group 0
    for g in scene.spec.geoms:
        if g.group in (1, 2):
            g.group = 0
    # the worker's clothing meshes are vertex lists inside the model; the converter
    # wants every mesh in a file
    for me in scene.spec.meshes:
        if not me.file and len(me.uservert):
            me.file = str(write_obj(assets / f"{me.name.replace('/', '__')}.obj", me.uservert, me.userface))
            me.uservert, me.userface = [], []
    xml = scene.spec.to_xml()
    # every file the model refers to ends up in the assets folder under its own name:
    # our textures and meshes are written there with absolute paths, the robot's
    # meshes come from the Menagerie folder with relative names, and one compiler
    # meshdir and texturedir cover both
    menagerie = MENAGERIE / "universal_robots_ur5e" / "assets"
    def relocate(match):
        ref = Path(match.group(1))
        if not ref.is_absolute():
            src = menagerie / ref.name
            if src.exists():
                shutil.copy(src, assets / ref.name)
        return f'file="{ref.name}"'
    xml = re.sub(r'file="([^"]+)"', relocate, xml)
    # dm_control's MJCF reader, which Gazebo's converter uses, reserves "/" in names,
    # and every attached part is named "worker/..." or "ur5e/...": a double underscore
    # takes its place in every attribute value that is not a directory
    def unslash(match):
        key, value = match.group(1), match.group(2)
        if key in ("meshdir", "texturedir", "file"):
            return match.group(0)
        return f'{key}="{value.replace("/", "__")}"'
    xml = re.sub(r'(\w+)="([^"]*)"', unslash, xml)
    compiler = f'<compiler meshdir="{assets.resolve()}" texturedir="{assets.resolve()}"'
    if "<compiler" in xml:
        xml = xml.replace("<compiler", compiler, 1)
    else:
        xml = xml.replace("<mujoco", "<mujoco", 1).replace(">", ">\n  " + compiler + "/>", 1)
    path = out / "sightline_cell.xml"
    path.write_text(xml)
    return path

