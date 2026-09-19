"""Convert the exported cell (station.export_mjcf) to an SDF world with Gazebo's own
converter, sdformat-mjcf, and the shims it needs on this machine.

    python ros2/mjcf_to_sdf.py results/ros2/scene/sightline_cell.xml results/ros2/gazebo/sightline_cell.sdf

The PyPI release of the converter (0.1.2) was written against the sdformat 13 binding
and gz.math 7; ROS Lyrical vendors sdformat 16 and gz.math under their plain names, and
the calls the converter makes are the same. gz.math must be imported before sdformat,
or the binding fails to start. The binding's setters want plain floats where the
MJCF reader hands them one element arrays. Free joints are skipped with a warning; the
replay world does not use joints.
"""
import sys
import time

import gz.math  # noqa: F401  (registers the types the sdformat binding needs, first)
import sdformat
import numpy as np

sys.modules["sdformat13"] = sdformat
for name in ("gz.math7", "gz.math8"):
    sys.modules[name] = gz.math


def plain_floats(cls):
    """Wrap every setter of a binding class so one element arrays arrive as floats."""
    for name in dir(cls):
        if not name.startswith("set_"):
            continue
        method = getattr(cls, name)

        def wrapped(self, *args, _m=method):
            args = tuple(float(a.item()) if isinstance(a, np.ndarray) and a.size == 1 else a for a in args)
            return _m(self, *args)
        setattr(cls, name, wrapped)


for cls_name in ("JointAxis", "Joint", "Link", "Inertial", "Geometry", "Visual", "Collision", "Material",
                 "Light", "Camera"):
    if hasattr(sdformat, cls_name):
        plain_floats(getattr(sdformat, cls_name))

from sdformat_mjcf.mjcf_to_sdformat.mjcf_to_sdformat import mjcf_file_to_sdformat  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    t0 = time.time()
    mjcf_file_to_sdformat(src, dst, export_world_plugins=True)
    print(f"converted {src} to {dst} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
