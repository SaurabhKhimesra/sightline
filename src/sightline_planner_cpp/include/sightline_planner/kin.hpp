// Kinematics on the planner's own robot model.
//
// The model holds the robot, its tool and its cameras and nothing else, so nothing
// here can read the person. It is the same model the cell was built from, which is
// what a real controller has in its URDF.
#ifndef SIGHTLINE_PLANNER_KIN_HPP
#define SIGHTLINE_PLANNER_KIN_HPP

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

#include "sightline_planner/types.hpp"

namespace sightline {

// Half length of the centre line and the radius around it, for one collision shape.
// A box is a point with the radius of its corners. Shapes with a centre line, a
// capsule or a cylinder, keep half their length in size[1].
std::pair<double, double> shape_line(const mjModel *m, int g);

// Forward kinematics, the tool tip Jacobian, and a damped least squares step.
class ToolKinematics {
 public:
  ToolKinematics(const mjModel *model, const std::string &site = "tool_tip",
                 const std::string &camera = "wrist_cam");
  ~ToolKinematics();

  ToolKinematics(const ToolKinematics &) = delete;
  ToolKinematics &operator=(const ToolKinematics &) = delete;

  const mjModel *model() const { return m_; }
  mjData *data() const { return d_; }
  int site_id() const { return site_; }
  int camera_id() const { return cam_; }
  bool limited(int j) const { return limited_[j]; }
  double qlim_low(int j) const { return qlim_(j, 0); }
  double qlim_high(int j) const { return qlim_(j, 1); }

  // Tool tip position and rotation for these joint angles.
  std::pair<Vec3, Mat3> fk(const Vec6 &q) const;

  // Where the wrist camera is, and which way it looks.
  //
  // mj_kinematics moves bodies and sites but not cameras, so this needs mj_camlight
  // as well. Without it the pose stays at the origin with a zero rotation, and every
  // hole lands behind the camera.
  std::pair<Vec3, Mat3> camera_pose(const Vec6 &q) const;

  double camera_fovy() const;

  // 6 x 6 tool tip Jacobian: linear rows then angular rows.
  Mat6 jacobian(const Vec6 &q) const;

  // Joint velocity that moves the tool tip toward a pose. Also gives the position error.
  // A null target rotation means the orientation is free.
  Vec6 velocity_to(const Vec6 &q, const Vec3 &target_pos, const Mat3 *target_R,
                   double *error_out, double gain_pos = 2.0, double gain_rot = 2.0,
                   double max_tool_speed = 0.25,
                   double max_joint_speed = 90.0 * M_PI / 180.0,
                   double damping = 0.05) const;

  Vec6 clamp(const Vec6 &q) const;

  // Same pose, fewest turns: of the whole-turn copies of each joint that lie inside
  // its limits, take the one closest to where the joint is now.
  //
  // Never clamps. The first version shifted a joint past its limit and then clamped
  // it, which is a different pose: a taught feeder pose moved 397 mm, onto the jig.
  Vec6 nearest(const Vec6 &q_goal, const Vec6 &q_now) const;

  // Damped least squares solve, starting from a seed. Also gives the position error.
  Vec6 ik(const Vec3 &target_pos, const Mat3 *target_R, const Vec6 &seed,
          double *error_out, int iters = 200) const;

  // Solve from several starts and keep the answer that moves the joints least.
  //
  // One seed and a straight velocity step is not enough: driving the tool along a
  // line walks joints into their limits and stalls there, which is what a real
  // controller avoids by planning the joint move first.
  Vec6 ik_multi(const Vec3 &target_pos, const Mat3 *target_R, const Vec6 &q_now,
                double *error_out, int iters = 300) const;

  // Every distinct way the arm can put the tool there: the answers from the same
  // starts as ik_multi and any extra ones, each shifted to the whole-turn copy
  // nearest q_ref, nearest first. Two answers are the same arm configuration if no
  // joint differs by more than apart.
  std::vector<Vec6> ik_all(const Vec3 &target_pos, const Mat3 *target_R, const Vec6 &q_ref,
                           const std::vector<Vec6> &extra_seeds = {}, int iters = 300,
                           double apart = 0.3) const;

 private:
  std::vector<Vec6> seed_set(const Vec6 &first, const std::vector<Vec6> &extra) const;

  const mjModel *m_;
  mjData *d_;
  int site_ = -1;
  int cam_ = -1;
  Eigen::Matrix<double, kNq, 2> qlim_;
  std::array<bool, kNq> limited_{};
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_KIN_HPP
