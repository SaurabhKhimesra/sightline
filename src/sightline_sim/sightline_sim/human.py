"""Two measured poses of the worker model, carried over from the author's earlier
cell-safety project (CobotSafe), where they were fitted on the same MuJoCo humanoid:
the arms hanging at rest, and the right arm reaching straight forward and down.
Joint angles in radians, by joint name.
"""

ARMS_AT_REST = {
    "lhumerusrx": -0.385, "lhumerusry": 0.271, "lhumerusrz": 1.348,
    "rhumerusrx": -0.856, "rhumerusry": -1.072, "rhumerusrz": -0.665,
}

RIGHT_ARM_REACH = {"rhumerusrx": 0.888, "rhumerusry": 1.178, "rhumerusrz": -0.721}
