"""Mesh builder invariants: closed solids wound outward, and profiles that triangulate."""

from __future__ import annotations

import unittest

import numpy as np

from sightline_sim import geometry as G
from sightline_sim.station import tslot_profile


def signed_volume(mesh: G.Mesh) -> float:
    """Volume of a closed mesh. Positive when the faces wind outward."""
    a, b, c = mesh.v[mesh.f[:, 0]], mesh.v[mesh.f[:, 1]], mesh.v[mesh.f[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


class TestSolids(unittest.TestCase):
    def test_cylinder_and_tube_wind_outward(self):
        self.assertGreater(signed_volume(G.cylinder(0.05, 0.2)), 0)
        tube = G.tube(0.05, 0.03, 0.2)
        ring = 0.2 * np.pi * (0.05 ** 2 - 0.03 ** 2)
        self.assertGreater(signed_volume(tube), 0.9 * ring)
        self.assertLess(signed_volume(tube), 1.1 * ring)

    def test_prism_and_limb_wind_outward(self):
        self.assertGreater(signed_volume(G.prism(G.rounded_rect(0.2, 0.1, 0.02), 0.0, 0.05)), 0)
        self.assertGreater(signed_volume(G.limb([0, 0, 0], [0, 0.3, 0], [0.05, 0.04, 0.03])), 0)

    def test_tslot_profile_triangulates(self):
        """Axis aligned profiles put vertices exactly on ear diagonals; ear clipping must cope."""
        mesh = G.prism(tslot_profile(0.040), 0.0, 0.5)
        self.assertGreater(len(mesh.f), 100)
        self.assertTrue(np.isfinite(mesh.v).all())
        self.assertGreater(signed_volume(mesh), 0)


class TestPlates(unittest.TestCase):
    def test_plate_with_holes_has_hole_walls(self):
        holes = [(sx * 0.09, sy * 0.06, G.circle(0.0023, 20)) for sx in (-1, 1) for sy in (-1, 1)]
        plate, walls = G.plate_with_holes(0.2, 0.15, 0.004, holes, corner_r=0.01)
        self.assertIsNotNone(walls)
        self.assertGreaterEqual(len(walls.f), 4 * 20 * 2)
        solid = G.prism(G.rounded_rect(0.2, 0.15, 0.01), -0.002, 0.002)
        self.assertGreater(signed_volume(plate), 0)
        self.assertLess(signed_volume(plate), signed_volume(solid))

    def test_slot_outline_runs_counter_clockwise(self):
        loop = G.stadium(0.012, 0.0053)
        self.assertGreaterEqual(len(loop), 20)
        self.assertGreater(G.signed_area(loop), 0)


if __name__ == "__main__":
    unittest.main()
