#include "sightline_planner/kin.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/LU>

namespace sightline {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The seeds every multi start solve uses, spread over the arm's configurations.
std::vector<Vec6> configuration_seeds() {
  std::vector<Vec6> out;
  for (double pan : {-M_PI / 2, 0.0, M_PI / 2, M_PI}) {
    for (double lift : {-2.2, -1.2}) {
      for (double elbow : {-1.8, 1.8}) {
        Vec6 s;
        s << pan, lift, elbow, -1.6, -1.57, 0.0;
        out.push_back(s);
      }
    }
  }
  return out;
}

Mat3 mat3_from_row_major(const mjtNum *m) {
  Mat3 out;
  out << m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8];
  return out;
}

}  // namespace

std::pair<double, double> shape_line(const mjModel *m, int g) {
  const int kind = m->geom_type[g];
  if (kind == mjGEOM_CAPSULE || kind == mjGEOM_CYLINDER) {
    return {m->geom_size[3 * g + 1], m->geom_size[3 * g + 0]};
  }
  if (kind == mjGEOM_BOX) {
    const Vec3 size(m->geom_size[3 * g + 0], m->geom_size[3 * g + 1], m->geom_size[3 * g + 2]);
    return {0.0, size.norm()};
  }
  return {0.0, m->geom_size[3 * g + 0]};
}

ToolKinematics::ToolKinematics(const mjModel *model, const std::string &site,
                               const std::string &camera)
    : m_(model), d_(mj_makeData(model)) {
  site_ = mj_name2id(model, mjOBJ_SITE, site.c_str());
  cam_ = mj_name2id(model, mjOBJ_CAMERA, camera.c_str());
  for (int j = 0; j < kNq; ++j) {
    qlim_(j, 0) = model->jnt_range[2 * j];
    qlim_(j, 1) = model->jnt_range[2 * j + 1];
    limited_[j] = model->jnt_limited[j] != 0;
  }
}

ToolKinematics::~ToolKinematics() {
  if (d_ != nullptr) {
    mj_deleteData(d_);
  }
}

std::pair<Vec3, Mat3> ToolKinematics::fk(const Vec6 &q) const {
  for (int j = 0; j < kNq; ++j) {
    d_->qpos[j] = q[j];
  }
  mj_kinematics(m_, d_);
  const Vec3 pos(d_->site_xpos[3 * site_], d_->site_xpos[3 * site_ + 1], d_->site_xpos[3 * site_ + 2]);
  return {pos, mat3_from_row_major(&d_->site_xmat[9 * site_])};
}

std::pair<Vec3, Mat3> ToolKinematics::camera_pose(const Vec6 &q) const {
  for (int j = 0; j < kNq; ++j) {
    d_->qpos[j] = q[j];
  }
  mj_kinematics(m_, d_);
  mj_camlight(m_, d_);
  const Vec3 pos(d_->cam_xpos[3 * cam_], d_->cam_xpos[3 * cam_ + 1], d_->cam_xpos[3 * cam_ + 2]);
  return {pos, mat3_from_row_major(&d_->cam_xmat[9 * cam_])};
}

double ToolKinematics::camera_fovy() const { return m_->cam_fovy[cam_]; }

Mat6 ToolKinematics::jacobian(const Vec6 &q) const {
  for (int j = 0; j < kNq; ++j) {
    d_->qpos[j] = q[j];
  }
  mj_kinematics(m_, d_);
  mj_comPos(m_, d_);
  std::vector<mjtNum> jp(3 * m_->nv, 0.0);
  std::vector<mjtNum> jr(3 * m_->nv, 0.0);
  mj_jacSite(m_, d_, jp.data(), jr.data(), site_);
  Mat6 out;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < kNq; ++col) {
      out(row, col) = jp[row * m_->nv + col];
      out(row + 3, col) = jr[row * m_->nv + col];
    }
  }
  return out;
}

Vec6 ToolKinematics::velocity_to(const Vec6 &q, const Vec3 &target_pos, const Mat3 *target_R,
                                 double *error_out, double gain_pos, double gain_rot,
                                 double max_tool_speed, double max_joint_speed,
                                 double damping) const {
  const auto [pos, R] = fk(q);
  const Vec3 e_pos = target_pos - pos;
  const double err = e_pos.norm();
  if (error_out != nullptr) {
    *error_out = err;
  }
  Vec3 v = gain_pos * e_pos;
  const double speed = v.norm();
  if (speed > max_tool_speed) {
    v *= max_tool_speed / speed;
  }
  Vec3 w = Vec3::Zero();
  if (target_R != nullptr) {
    const Mat3 dR = (*target_R) * R.transpose();
    mjtNum mat[9];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        mat[row * 3 + col] = dR(row, col);
      }
    }
    mjtNum quat[4];
    mju_mat2Quat(quat, mat);
    const Vec3 imag(quat[1], quat[2], quat[3]);
    const double angle = 2.0 * std::atan2(imag.norm(), quat[0]);
    const Vec3 axis = imag / std::max(imag.norm(), 1e-9);
    w = gain_rot * (angle > M_PI ? angle - 2 * M_PI : angle) * axis;
  }
  const Mat6 J = jacobian(q);
  Eigen::Matrix<double, 6, 1> task;
  task.head<3>() = v;
  task.tail<3>() = w;
  const Mat6 damped = J * J.transpose() + damping * damping * Mat6::Identity();
  Vec6 qd = J.transpose() * damped.lu().solve(task);
  const double fastest = qd.cwiseAbs().maxCoeff();
  if (fastest > max_joint_speed) {
    qd *= max_joint_speed / fastest;
  }
  return qd;
}

Vec6 ToolKinematics::clamp(const Vec6 &q) const {
  Vec6 out = q;
  for (int j = 0; j < kNq; ++j) {
    if (limited_[j]) {
      out[j] = std::min(std::max(out[j], qlim_(j, 0)), qlim_(j, 1));
    }
  }
  return out;
}

Vec6 ToolKinematics::nearest(const Vec6 &q_goal, const Vec6 &q_now) const {
  Vec6 out = q_goal;
  for (int j = 0; j < kNq; ++j) {
    double best = out[j];
    double best_gap = kInf;
    bool any = false;
    for (int k = -2; k <= 2; ++k) {
      const double candidate = out[j] + k * 2 * M_PI;
      if (limited_[j] && (candidate < qlim_(j, 0) - 1e-9 || candidate > qlim_(j, 1) + 1e-9)) {
        continue;
      }
      const double gap = std::abs(candidate - q_now[j]);
      if (!any || gap < best_gap) {
        best = candidate;
        best_gap = gap;
        any = true;
      }
    }
    if (any) {
      out[j] = best;
    }
  }
  return out;
}

Vec6 ToolKinematics::ik(const Vec3 &target_pos, const Mat3 *target_R, const Vec6 &seed,
                        double *error_out, int iters) const {
  Vec6 q = seed;
  double err = kInf;
  for (int i = 0; i < iters; ++i) {
    const Vec6 qd = velocity_to(q, target_pos, target_R, &err, 1.0, 1.0, 1e9, 1e9, 0.02);
    if (err < 1e-5) {
      break;
    }
    q = clamp(q + 0.5 * qd);
  }
  if (error_out != nullptr) {
    *error_out = err;
  }
  return q;
}

std::vector<Vec6> ToolKinematics::seed_set(const Vec6 &first,
                                           const std::vector<Vec6> &extra) const {
  std::vector<Vec6> seeds{first};
  seeds.insert(seeds.end(), extra.begin(), extra.end());
  const std::vector<Vec6> grid = configuration_seeds();
  seeds.insert(seeds.end(), grid.begin(), grid.end());
  return seeds;
}

Vec6 ToolKinematics::ik_multi(const Vec3 &target_pos, const Mat3 *target_R, const Vec6 &q_now,
                              double *error_out, int iters) const {
  Vec6 best_q = q_now;
  double best_err = kInf;
  double best_cost = kInf;
  bool found = false;
  for (const Vec6 &seed : seed_set(q_now, {})) {
    double err = kInf;
    Vec6 q = ik(target_pos, target_R, seed, &err, iters);
    if (err > 1e-4) {
      continue;
    }
    q = nearest(q, q_now);
    q = ik(target_pos, target_R, q, &err, 60);   // the shift may need a nudge back
    if (err > 1e-4) {
      continue;
    }
    const double cost = (q - q_now).cwiseAbs().maxCoeff();
    if (cost < best_cost) {
      best_q = q;
      best_err = err;
      best_cost = cost;
      found = true;
    }
  }
  if (error_out != nullptr) {
    *error_out = found ? best_err : kInf;
  }
  return found ? best_q : q_now;
}

std::vector<Vec6> ToolKinematics::ik_all(const Vec3 &target_pos, const Mat3 *target_R,
                                         const Vec6 &q_ref, const std::vector<Vec6> &extra_seeds,
                                         int iters, double apart) const {
  std::vector<Vec6> found;
  for (const Vec6 &seed : seed_set(q_ref, extra_seeds)) {
    double err = kInf;
    Vec6 q = ik(target_pos, target_R, seed, &err, iters);
    if (err > 1e-4) {
      continue;
    }
    q = nearest(q, q_ref);
    q = ik(target_pos, target_R, q, &err, 60);
    if (err > 1e-4) {
      continue;
    }
    bool distinct = true;
    for (const Vec6 &other : found) {
      if ((q - other).cwiseAbs().maxCoeff() <= apart) {
        distinct = false;
        break;
      }
    }
    if (distinct) {
      found.push_back(q);
    }
  }
  std::sort(found.begin(), found.end(), [&q_ref](const Vec6 &a, const Vec6 &b) {
    return (a - q_ref).cwiseAbs().maxCoeff() < (b - q_ref).cwiseAbs().maxCoeff();
  });
  return found;
}

}  // namespace sightline
