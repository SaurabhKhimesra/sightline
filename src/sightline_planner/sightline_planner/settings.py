"""The numbers the cell and the planner share: rates, the eyes picture, the
calibration error the planner lives with, and two scene choices about waiting.
"""

FPS = 25.0               # the eyes camera, the grid, the judge's R2 frames and the video (a design choice)
MOTION_HZ = 100.0
CAM_W, CAM_H = 640, 480
CALIB_POS_MM, CALIB_DEG = 2.0, 0.15
HOLD_ASK_S = 2.0         # held this long by the person: ask him to move his hand (scene choice)
HELD_FRACTION = 0.10     # moving slower than this share of what the task asked counts as held
# A blind cell can hold his hand only this close to a seen part of him, a hand's length
# (scene choice): a hidden hand hangs on a wrist and forearm, and behind the frame's
# uprights and beams those stay in view. Counted from the grid's cells, which drop
# stray noise, not from raw points: with raw points and 35 cm the cells beside the
# feeder were live in every picture, from 2 to 10 stray pixels on the worktop, and the
# arm could not reach its own feeder. With the grid and 20 cm they are live in 15 %.
HAND_REACH_M = 0.20
