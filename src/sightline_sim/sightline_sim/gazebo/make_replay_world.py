"""Build the Gazebo world that replays a recorded run.

Gazebo cannot read a MuJoCo model and has no way to play back recorded body poses,
so the cell goes through Gazebo's own converter (mjcf2sdf, on the MJCF that
station.export_mjcf writes) and this script turns the result into a replay world:
every MJCF body that has something to see becomes its own model, so its pose can be
set from the recording every frame through the set_pose_vector service; physics has
no gravity and no collisions; the two cameras that matter are sensors at the poses
MuJoCo used. Rendering only: the physics, the worker and the judge stay in MuJoCo.

    python ros2/make_replay_world.py results/ros2/scene/sightline_cell.xml \\
        results/ros2/gazebo/sightline_cell.sdf results/ros2/gazebo/sightline_replay.sdf
"""
import os
import re
import sys

import mujoco
import numpy as np

SDF_FROM_MJ_CAMERA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])   # columns: sdf x, y, z in mj camera axes
CAMERAS = {"cycle_hero": (1280, 720), "eyes_cam": (960, 720)}


# cameras the replay adds, placed here in world coordinates: name -> (position, look at, fovy deg, (w, h))
EXTRA_CAMERAS = {
    # from the far side of the bench, behind the robot: his face and hands, the arm and
    # the box in one frame (the user's ask, 2026-09-19); the hero camera sees his back
    "front": ((-0.40, 2.50, 2.25), (-0.15, -0.40, 1.20), 42.0, (1280, 720)),
}
SHADOW_LIGHTS = ("key_l", "key_r")    # the lights that cast shadows in the sensors' render, when asked


def look_at_pose(pos, target):
    """An SDF camera pose (x y z roll pitch yaw) looking from pos at target: the camera
    looks down its +x with z up, so yaw turns it toward the target and pitch tips it."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    yaw = np.arctan2(f[1], f[0])
    pitch = np.arctan2(-f[2], np.hypot(f[0], f[1]))
    return f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} 0 {pitch:.5f} {yaw:.5f}"


def rpy(R):
    """ZYX roll, pitch, yaw of a rotation matrix, as SDF poses want them."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    c = np.cos(pitch)
    if abs(c) < 1e-9:
        return 0.0, float(pitch), float(np.arctan2(-R[0, 1], R[1, 1]))
    return float(np.arctan2(R[2, 1] / c, R[2, 2] / c)), float(pitch), float(np.arctan2(R[1, 0] / c, R[0, 0] / c))


def pose_text(pos, R):
    r, p, y = rpy(R)
    return f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {r:.6f} {p:.6f} {y:.6f}"


def build(mjcf_path, sdf_in, sdf_out, shadows=False, gui_camera="cycle_hero"):
    m = mujoco.MjModel.from_xml_path(mjcf_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mujoco.mj_camlight(m, d)
    s = open(sdf_in).read()
    # the links, with their visuals verbatim: link frame is the MJCF body frame
    models = []
    for match in re.finditer(r"<link name='([^']+)'>(.*?)</link>", s, re.S):
        name, body = match.group(1), match.group(2)
        visuals = re.findall(r"<visual\b.*?</visual>", body, re.S)
        if not visuals:
            continue
        try:
            b = m.body(name).id
        except KeyError:
            if name != "link_0":
                continue                   # dummy links of multi axis joints, nothing to see
            b = 0                          # the converter's name for the world body: the floor
        R = d.xmat[b].reshape(3, 3)
        models.append(f"    <model name='{name}'>\n      <pose>{pose_text(d.xpos[b], R)}</pose>\n"
                      f"      <link name='body'>\n        " + "\n        ".join(visuals) + "\n      </link>\n    </model>")
    # the cameras, as MuJoCo placed them, in SDF's optical convention
    sensors = []
    for cam, (w, h) in CAMERAS.items():
        c = m.cam(cam).id
        R = d.cam_xmat[c].reshape(3, 3) @ SDF_FROM_MJ_CAMERA
        fovy = np.radians(float(m.cam_fovy[c]))
        hfov = 2 * np.arctan(np.tan(fovy / 2) * w / h)
        sensors.append(f"""      <sensor name='{cam}' type='camera'>
        <pose>{pose_text(d.cam_xpos[c], R)}</pose>
        <update_rate>25</update_rate><always_on>1</always_on><visualize>0</visualize>
        <topic>/replay/{cam}</topic>
        <camera><horizontal_fov>{hfov:.6f}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>RGB_INT8</format><anti_aliasing>4</anti_aliasing></image>
          <clip><near>0.05</near><far>40</far></clip>
        </camera>
      </sensor>""")
    # the desktop GUI's view, where the hero camera is, with its lens: written as a
    # config file for `gz sim -g --gui-config`. Not <shadows>: the GUI crashes on it
    # while a server is up
    if gui_camera in EXTRA_CAMERAS:
        pos, target, fovy, (w, h) = EXTRA_CAMERAS[gui_camera]
        gui_pose = look_at_pose(pos, target)
    else:
        hero = m.camera(gui_camera).id
        gui_pose = pose_text(d.cam_xpos[hero], d.cam_xmat[hero].reshape(3, 3) @ SDF_FROM_MJ_CAMERA)
        fovy, (w, h) = float(m.cam_fovy[hero]), CAMERAS[gui_camera]
    gui_hfov = np.degrees(2 * np.arctan(np.tan(np.radians(fovy) / 2) * w / h))
    hidden = ("<gz-gui><property key='state' type='string'>floating</property>"
              "<property key='width' type='double'>5</property><property key='height' type='double'>5</property>"
              "<property key='showTitleBar' type='bool'>false</property></gz-gui>")
    gui = f"""    <gui fullscreen='false'>
      <plugin filename='MinimalScene' name='3D View'>
        <gz-gui><title>3D View</title><property type='bool' key='showTitleBar'>false</property>
          <property type='string' key='state'>docked</property></gz-gui>
        <engine>ogre2</engine><scene>scene</scene><ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.55 0.58 0.62</background_color>
        <camera_pose>{gui_pose}</camera_pose>
        <horizontal_fov>{gui_hfov:.2f}</horizontal_fov>
      </plugin>
      <plugin filename='GzSceneManager' name='Scene Manager'>{hidden}</plugin>
      <plugin filename='InteractiveViewControl' name='Interactive view control'>{hidden}</plugin>
      <plugin filename='MarkerManager' name='Marker manager'>{hidden}</plugin>
      <plugin filename='VisualizationCapabilities' name='Visualization Capabilities'>{hidden}</plugin>
    </gui>
"""
    # a light and a model may not share a name in Gazebo, and the hall's light is
    # named after the hall, so every light gets a prefix; and no light visuals in the
    # GUI (green pyramids at every spot light)
    for cam, (pos, target, fovy, (w, h)) in EXTRA_CAMERAS.items():
        hfov = 2 * np.arctan(np.tan(np.radians(fovy) / 2) * w / h)
        sensors.append(f"""      <sensor name='{cam}' type='camera'>
        <pose>{look_at_pose(pos, target)}</pose>
        <topic>/replay/{cam}</topic>
        <update_rate>25</update_rate><always_on>1</always_on><visualize>0</visualize>
        <camera>
          <horizontal_fov>{hfov:.5f}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>RGB_INT8</format><anti_aliasing>4</anti_aliasing></image>
          <clip><near>0.05</near><far>40</far></clip>
        </camera>
      </sensor>""")
    lights = [re.sub(r"(<light name='[^']*' type='[^']*'>)", r"\1<visualize>false</visualize>",
                     re.sub(r"<light name='", "<light name='light_", light, count=1), count=1)
              for light in re.findall(r"<light\b.*?</light>", s, re.S)]
    if shadows:
        lights = [light.replace("<cast_shadows>false</cast_shadows>", "<cast_shadows>true</cast_shadows>")
                  if any(f"<light name='light_{n}'" in light for n in SHADOW_LIGHTS) else light for light in lights]
    scene = re.search(r"<scene>.*?</scene>", s, re.S)
    world = f"""<?xml version='1.0'?>
<sdf version='1.10'>
  <world name='sightline_replay'>
    <plugin name='gz::sim::systems::Physics' filename='gz-sim-physics-system'/>
    <plugin name='gz::sim::systems::Sensors' filename='gz-sim-sensors-system'><render_engine>ogre2</render_engine></plugin>
    <plugin name='gz::sim::systems::UserCommands' filename='gz-sim-user-commands-system'/>
    <plugin name='gz::sim::systems::SceneBroadcaster' filename='gz-sim-scene-broadcaster-system'/>
    <gravity>0 0 0</gravity>
    <physics type='ode'><max_step_size>0.001</max_step_size><real_time_factor>0</real_time_factor></physics>
    {scene.group(0) if scene else ''}
    {chr(10).join('    ' + light for light in lights)}
    <model name='cameras'><static>true</static><pose>0 0 0 0 0 0</pose><link name='link'>
{chr(10).join(sensors)}
    </link></model>
{chr(10).join(models)}
  </world>
</sdf>
"""
    open(sdf_out, "w").write(world)
    # the same view as a GUI config file, for a desktop Gazebo started with --gui-config:
    # only the 3D view and the world stats, the size of the recording tile
    plugins = [line for line in gui.splitlines() if "<gui " not in line and "</gui>" not in line]
    config = f"""<?xml version='1.0'?>
<dialog name='quick_start' show_again='false'/>
<window>
  <width>1280</width><height>720</height>
  <style material_theme='Light' material_primary='DeepOrange' material_accent='LightBlue'/>
  <menus><drawer default='false'></drawer></menus>
  <dialog_on_exit>false</dialog_on_exit>
</window>
{chr(10).join(plugins)}
"""
    config_path = os.path.join(os.path.dirname(sdf_out), "gazebo_gui.config")
    open(config_path, "w").write(config)
    print(f"{len(models)} body models, {len(lights)} lights, cameras {list(CAMERAS) + list(EXTRA_CAMERAS)}"
          f"{' with shadows from ' + ', '.join(SHADOW_LIGHTS) if shadows else ''}, GUI on {gui_camera}; "
          f"wrote {sdf_out} and {config_path}")


def main():
    build(*sys.argv[1:4], shadows="--shadows" in sys.argv,
          gui_camera=sys.argv[sys.argv.index("--gui-camera") + 1] if "--gui-camera" in sys.argv else "cycle_hero")


if __name__ == "__main__":
    main()
