// The parts of the planner that need no cell model: the QP solver and the pieces of
// geometry the motion layer and the grid are built on.
#include <cmath>

#include <gtest/gtest.h>

#include "sightline_planner/motion.hpp"
#include "sightline_planner/perceive.hpp"
#include "sightline_planner/qp.hpp"

using sightline::MatX;
using sightline::VecX;

// The unconstrained minimum, when no constraint is in the way.
TEST(Qp, FindsTheUnconstrainedMinimum) {
  MatX G = MatX::Identity(2, 2);
  VecX a(2);
  a << -1.0, -2.0;
  const sightline::qp::Solution sol = sightline::qp::solve(G, a, MatX::Zero(2, 0), VecX::Zero(0));
  ASSERT_TRUE(sol.feasible);
  EXPECT_NEAR(sol.x[0], 1.0, 1e-9);
  EXPECT_NEAR(sol.x[1], 2.0, 1e-9);
}

// One constraint biting, and the optimality conditions met at the answer.
TEST(Qp, KeepsATightConstraint) {
  MatX G = MatX::Identity(2, 2);
  VecX a(2);
  a << 0.0, 0.0;
  MatX C(2, 1);
  C << 1.0, 0.0;              // x0 >= 1
  VecX b(1);
  b << 1.0;
  const sightline::qp::Solution sol = sightline::qp::solve(G, a, C, b);
  ASSERT_TRUE(sol.feasible);
  EXPECT_NEAR(sol.x[0], 1.0, 1e-9);
  EXPECT_NEAR(sol.x[1], 0.0, 1e-9);
  const sightline::qp::KktError err = sightline::qp::kkt_error(G, a, C, b, sol);
  EXPECT_LT(err.stationarity, 1e-9);
  EXPECT_LT(err.primal, 1e-9);
  EXPECT_LT(err.dual, 1e-9);
  EXPECT_LT(err.complementarity, 1e-9);
}

// The damper lets a pair close only as fast as it could still stop.
TEST(Motion, ApproachLimitIsZeroAtTheSafetyDistance) {
  EXPECT_NEAR(sightline::approach_limit(0.10, 0.10), 0.0, 1e-12);
  EXPECT_GT(sightline::approach_limit(0.30, 0.10), 0.0);
}

// The closest point on a segment, which every distance in the motion layer uses.
TEST(Motion, SegmentDistanceClampsToTheEnds) {
  sightline::Vec3 closest;
  double dist = 0.0;
  sightline::segment_distance(sightline::Vec3(2.0, 1.0, 0.0), sightline::Vec3(0.0, 0.0, 0.0),
                              sightline::Vec3(1.0, 0.0, 0.0), &closest, &dist);
  EXPECT_NEAR(closest[0], 1.0, 1e-12);
  EXPECT_NEAR(dist, std::sqrt(2.0), 1e-12);
}

// Voxel keys survive a round trip, including negative indices.
TEST(Perceive, VoxelKeysRoundTrip) {
  int x = 0;
  int y = 0;
  int z = 0;
  sightline::unpack(sightline::pack(-3, 17, -1), &x, &y, &z);
  EXPECT_EQ(x, -3);
  EXPECT_EQ(y, 17);
  EXPECT_EQ(z, -1);
}
