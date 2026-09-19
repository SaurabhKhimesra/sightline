"""Move and resize an X window, or find the windows of a process: what wmctrl and
xdotool do, without either, through libX11 and xprop.

    ros2 run sightline_ros xwin move <window id> <x> <y> <width> <height>
    ros2 run sightline_ros xwin find <pid> [<pid> ...]        # top-level windows whose _NET_WM_PID is one of these

The request goes to mutter as an ordinary ConfigureRequest, which it honours for
XWayland clients (locrec's ros2/xresize.py, the same way).
"""
import ctypes
import re
import subprocess
import sys


def move(win: int, x: int, y: int, width: int, height: int) -> None:
    x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        sys.exit("cannot open the X display")
    x11.XMoveResizeWindow(display, win, x, y, width, height)
    x11.XFlush(display)
    x11.XCloseDisplay(display)


def find(pids) -> list:
    """(window id, width, height) of every named top-level window owned by one of the pids,
    largest first."""
    tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
    found = []
    for line in tree.splitlines():
        m = re.match(r"\s+(0x[0-9a-f]+) \"(.*)\".*?(\d+)x(\d+)[+-]", line)
        if not m or "has no name" in line:
            continue
        wid, w, h = m.group(1), int(m.group(3)), int(m.group(4))
        if w < 200 or h < 200:
            continue
        prop = subprocess.run(["xprop", "-id", wid, "_NET_WM_PID"], capture_output=True, text=True).stdout
        pm = re.search(r"= (\d+)", prop)
        if pm and int(pm.group(1)) in pids:
            found.append((wid, w, h))
    return sorted(found, key=lambda t: -t[1] * t[2])


def main():
    if sys.argv[1] == "move":
        move(int(sys.argv[2], 16), *map(int, sys.argv[3:7]))
    elif sys.argv[1] == "find":
        for wid, w, h in find({int(p) for p in sys.argv[2:]}):
            print(wid, w, h)


if __name__ == "__main__":
    main()
