#include "sightline_planner/motion.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

namespace sightline {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

std::string label(const char *what, int body, double dist_m) {
  std::ostringstream out;
  out << what << " body " << body << " at " << std::llround(dist_m * 1000.0) << " mm";
  return out.str();
}

}  // namespace

double approach_limit(double dist, double safe) {
  return std::sqrt(2.0 * kABrake * std::max(dist - safe, 0.0));
}

Points Obstacles::guard_points() const {
  if (unseen.rows() > 0 && person.rows() > 0) {
    Points out(person.rows() + unseen.rows(), 3);
    out.topRows(person.rows()) = person;
    out.bottomRows(unseen.rows()) = unseen;
    return out;
  }
  return person.rows() > 0 ? person : unseen;
}

void segment_distance(const Vec3 &p, const Vec3 &a, const Vec3 &b, Vec3 *closest,
                      double *dist) {
  const Vec3 ab = b - a;
  const double denom = ab.dot(ab);
  const double t = denom < 1e-12 ? 0.0 : std::min(std::max((p - a).dot(ab) / denom, 0.0), 1.0);
  const Vec3 point = a + t * ab;
  if (closest != nullptr) {
    *closest = point;
  }
  if (dist != nullptr) {
    *dist = (p - point).norm();
  }
}

void segment_to_segment(const Vec3 &a0, const Vec3 &a1, const Vec3 &b0, const Vec3 &b1,
                        Vec3 *on_a, Vec3 *on_b, double *dist) {
  double best = kInf;
  Vec3 best_a = a0;
  Vec3 best_b = b0;
  for (int k = 0; k < 8; ++k) {
    const double t = k / 7.0;
    const Vec3 p = a0 + t * (a1 - a0);
    Vec3 q;
    double gap = 0.0;
    segment_distance(p, b0, b1, &q, &gap);
    if (gap < best) {
      best = gap;
      best_a = p;
      best_b = q;
    }
  }
  if (on_a != nullptr) {
    *on_a = best_a;
  }
  if (on_b != nullptr) {
    *on_b = best_b;
  }
  if (dist != nullptr) {
    *dist = best;
  }
}

MotionLayer::MotionLayer(ToolKinematics *kin, bool use_r1, bool use_r2, double joint_speed,
                         double joint_accel, double dt)
    : kin_(kin),
      use_r1_(use_r1),
      use_r2_(use_r2),
      joint_speed_(joint_speed),
      joint_accel_(joint_accel),
      dt_(dt) {
  capsules_ = collision_shapes();
}

std::vector<MotionLayer::Capsule> MotionLayer::collision_shapes() const {
  const mjModel *m = kin_->model();
  std::vector<Capsule> out;
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_group[g] != 3) {
      continue;
    }
    const auto [half, radius] = shape_line(m, g);
    out.push_back(Capsule{g, m->geom_bodyid[g], half, radius});
  }
  return out;
}

std::vector<MotionLayer::WorldCapsule> MotionLayer::world_capsules(const Vec6 &q) const {
  const mjModel *m = kin_->model();
  mjData *d = kin_->data();
  for (int j = 0; j < kNq; ++j) {
    d->qpos[j] = q[j];
  }
  mj_kinematics(m, d);
  std::vector<WorldCapsule> out;
  out.reserve(capsules_.size());
  for (const Capsule &cap : capsules_) {
    const Vec3 centre(d->geom_xpos[3 * cap.geom], d->geom_xpos[3 * cap.geom + 1],
                      d->geom_xpos[3 * cap.geom + 2]);
    // the third column of the shape's rotation is its own axis
    const Vec3 axis(d->geom_xmat[9 * cap.geom + 2], d->geom_xmat[9 * cap.geom + 5],
                    d->geom_xmat[9 * cap.geom + 8]);
    out.push_back(WorldCapsule{cap.body, centre - axis * cap.half, centre + axis * cap.half,
                               cap.radius});
  }
  return out;
}

Eigen::Matrix<double, 3, 6> MotionLayer::point_jacobian(const Vec3 &point, int body) const {
  const mjModel *m = kin_->model();
  mjData *d = kin_->data();
  std::vector<mjtNum> jacp(3 * m->nv, 0.0);
  const mjtNum p[3] = {point[0], point[1], point[2]};
  mj_jac(m, d, jacp.data(), nullptr, p, body);
  Eigen::Matrix<double, 3, 6> out;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < kNq; ++col) {
      out(row, col) = jacp[row * m->nv + col];
    }
  }
  return out;
}

Report MotionLayer::solve(const Vec6 &q, const Eigen::Matrix<double, 6, 1> &twist,
                          const Obstacles &obstacles, const Vec6 *qd_prev) {
  const Vec6 previous = qd_prev == nullptr ? last_ : *qd_prev;
  const Mat6 J = kin_->jacobian(q);
  const Mat6 G = J.transpose() * J + (kSmooth + kContinuity) * Mat6::Identity();
  const Vec6 a = -(J.transpose() * twist) - kContinuity * previous;
  return solve_core(q, G, a, obstacles, previous);
}

Report MotionLayer::solve_joint(const Vec6 &q, const Vec6 &qd_task, const Obstacles &obstacles,
                                const Vec6 *qd_prev) {
  const Vec6 previous = qd_prev == nullptr ? last_ : *qd_prev;
  const Mat6 G = (1.0 + kContinuity) * Mat6::Identity();
  const Vec6 a = -qd_task - kContinuity * previous;
  return solve_core(q, G, a, obstacles, previous);
}

Report MotionLayer::solve_core(const Vec6 &q, const Mat6 &G, Vec6 a, const Obstacles &obstacles,
                               const Vec6 &qd_prev) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Vec6> rows;
  std::vector<double> bounds;
  labels_.clear();

  // joint speed, joint acceleration and the joint limits themselves
  for (int j = 0; j < kNq; ++j) {
    Vec6 e = Vec6::Zero();
    e[j] = 1.0;
    double lo_speed = std::max(-joint_speed_, qd_prev[j] - joint_accel_ * dt_);
    double hi_speed = std::min(joint_speed_, qd_prev[j] + joint_accel_ * dt_);
    if (kin_->limited(j)) {
      lo_speed = std::max(lo_speed, (kin_->qlim_low(j) - q[j]) / dt_);
      hi_speed = std::min(hi_speed, (kin_->qlim_high(j) - q[j]) / dt_);
    }
    if (lo_speed > hi_speed) {          // a limit and a ramp disagree: keep the limit
      const double lo = std::min(lo_speed, hi_speed);
      const double hi = std::max(lo_speed, hi_speed);
      lo_speed = hi_speed = std::min(std::max(0.0, lo), hi);
    }
    rows.push_back(e);
    bounds.push_back(lo_speed);
    rows.push_back(-e);
    bounds.push_back(-hi_speed);
    labels_.push_back("joint " + std::to_string(j) + " low");
    labels_.push_back("joint " + std::to_string(j) + " high");
  }

  const std::vector<WorldCapsule> caps = world_capsules(q);
  int r1_rows = 0;
  int r2_rows = 0;
  double closest_person = kInf;
  double closest_guard = kInf;
  const int n_seen = static_cast<int>(obstacles.person.rows());

  if (use_r1_) {
    const Points guard = obstacles.guard_points();
    if (guard.rows() > 0) {
      const Points &speeds = obstacles.person_velocity;
      for (const WorldCapsule &cap : caps) {
        for (const NearPoint &pair :
             near_points(guard, cap.p0, cap.p1, cap.radius, kDInfluence)) {
          closest_guard = std::min(closest_guard, pair.dist);
          if (pair.index < n_seen) {
            closest_person = std::min(closest_person, pair.dist);
          }
          Vec3 normal = pair.point - pair.witness;
          const double norm = normal.norm();
          if (norm < 1e-9) {
            continue;
          }
          normal /= norm;
          const Eigen::Matrix<double, 3, 6> jac = point_jacobian(pair.witness, cap.body);
          double moving = 0.0;
          if (pair.index < static_cast<int>(speeds.rows()) && speeds.rows() > 0) {
            moving = normal.dot(speeds.row(pair.index).transpose());
          }
          // d_dot = n.v_person - n.v_robot must stay above the damper's floor
          const Vec6 row = (normal.transpose() * jac).transpose();
          rows.push_back(-row);
          const double floor = -approach_limit(pair.dist, kDSafe) - moving;
          bounds.push_back(std::min(floor, 0.0));
          if (pair.dist < kDSafe) {          // too close already: ask to back out
            a += kRetreatWeight * (kDSafe - pair.dist) * row;
          }
          labels_.push_back(label("R1", cap.body, pair.dist));
          r1_rows += 1;
        }
      }
    }
  }

  if (use_r2_ && obstacles.has_sight_from && obstacles.person.rows() > 0) {
    const Vec3 eye = obstacles.sight_from;
    for (const WorldCapsule &cap : caps) {
      for (const NearLine &line :
           near_lines(obstacles.person, eye, cap.p0, cap.p1, cap.radius, kVisInfluence)) {
        Vec3 normal = line.on_line - line.witness;
        const double norm = normal.norm();
        if (norm < 1e-9) {
          continue;
        }
        normal /= norm;
        const Eigen::Matrix<double, 3, 6> jac = point_jacobian(line.witness, cap.body);
        const Vec6 row = (normal.transpose() * jac).transpose();
        rows.push_back(-row);
        bounds.push_back(std::min(-approach_limit(line.dist, kRVis), 0.0));
        if (line.dist < kRVis) {
          a += kRetreatWeight * (kRVis - line.dist) * row;
        }
        labels_.push_back(label("R2", cap.body, line.dist));
        r2_rows += 1;
      }
    }
  }

  for (const StaticSegment &seg : obstacles.statics) {
    for (const WorldCapsule &cap : caps) {
      Vec3 witness;
      Vec3 on_static;
      double dist = 0.0;
      segment_to_segment(cap.p0, cap.p1, seg.p0, seg.p1, &witness, &on_static, &dist);
      dist -= cap.radius + seg.radius;
      if (dist > kStaticInfluence) {
        continue;
      }
      Vec3 normal = on_static - witness;
      const double norm = normal.norm();
      if (norm < 1e-9) {
        continue;
      }
      normal /= norm;
      const Eigen::Matrix<double, 3, 6> jac = point_jacobian(witness, cap.body);
      const Vec6 row = (normal.transpose() * jac).transpose();
      rows.push_back(-row);
      bounds.push_back(std::min(-approach_limit(dist, kDSafeStatic), 0.0));
      if (dist < kDSafeStatic) {
        a += kRetreatWeight * (kDSafeStatic - dist) * row;
      }
      labels_.push_back(label("static", cap.body, dist));
    }
  }

  for (const Plane &plane : obstacles.planes) {
    const Vec3 normal = plane.normal / std::max(plane.normal.norm(), 1e-9);
    for (const WorldCapsule &cap : caps) {
      const double at0 = (cap.p0 - plane.point).dot(normal);
      const double at1 = (cap.p1 - plane.point).dot(normal);
      const double lowest = std::min(at0, at1) - cap.radius;
      if (lowest > kStaticInfluence) {
        continue;
      }
      const Vec3 witness = at0 <= at1 ? cap.p0 : cap.p1;
      const Eigen::Matrix<double, 3, 6> jac = point_jacobian(witness, cap.body);
      const Vec6 row = (normal.transpose() * jac).transpose();
      rows.push_back(row);       // speed along the normal must stay above the floor
      bounds.push_back(std::min(-approach_limit(lowest, kDSafeStatic), 0.0));
      if (lowest < kDSafeStatic) {
        a -= kRetreatWeight * (kDSafeStatic - lowest) * row;
      }
      labels_.push_back(label("plane", cap.body, lowest));
    }
  }

  MatX C(kNq, static_cast<Eigen::Index>(rows.size()));
  VecX b(static_cast<Eigen::Index>(bounds.size()));
  for (size_t i = 0; i < rows.size(); ++i) {
    C.col(static_cast<Eigen::Index>(i)) = rows[i];
    b[static_cast<Eigen::Index>(i)] = bounds[i];
  }

  qp::Solution sol = qp::solve(G, a, C, b);
  std::string fallback;
  if (!sol.feasible) {
    // the rules alone, with no task at all: move away and nothing else, and with
    // the acceleration a protective stop is allowed rather than the working ramp
    fallback = "rules only";
    VecX b2 = b;
    for (int j = 0; j < kNq; ++j) {
      double lo = std::max(-joint_speed_, qd_prev[j] - 3 * joint_accel_ * dt_);
      double hi = std::min(joint_speed_, qd_prev[j] + 3 * joint_accel_ * dt_);
      if (kin_->limited(j)) {
        lo = std::max(lo, (kin_->qlim_low(j) - q[j]) / dt_);
        hi = std::min(hi, (kin_->qlim_high(j) - q[j]) / dt_);
      }
      b2[2 * j] = lo;
      b2[2 * j + 1] = -hi;
    }
    sol = qp::solve(Mat6::Identity(), Vec6::Zero(), C, b2);
    if (!sol.feasible) {
      // slow every joint toward zero at three times the working deceleration.
      // The first version set the speed to minus the last one, clipped: that
      // does not brake, it throws each joint into reverse.
      fallback = "brake";
      const double step = 3 * joint_accel_ * dt_;
      Vec6 braked = qd_prev;
      for (int j = 0; j < kNq; ++j) {
        braked[j] = qd_prev[j] - std::min(std::max(qd_prev[j], -step), step);
      }
      sol = qp::Solution{braked, {}, VecX::Zero(0), 0, false};
    }
  }
  solution_ = sol;

  Vec6 qd;
  for (int j = 0; j < kNq; ++j) {
    qd[j] = std::min(std::max(sol.x[j], -joint_speed_), joint_speed_);
  }
  last_ = qd;

  Report report;
  report.qd = qd;
  report.constraints = static_cast<int>(bounds.size());
  report.r1_active = r1_rows;
  report.r2_active = r2_rows;
  report.fallback = fallback;
  report.min_person_distance = closest_person;
  report.min_guard_distance = closest_guard;
  report.solve_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return report;
}

std::vector<MotionLayer::NearPoint> MotionLayer::near_points(const Points &points, const Vec3 &p0,
                                                            const Vec3 &p1, double radius,
                                                            double influence) const {
  const Vec3 ab = p1 - p0;
  const double denom = ab.dot(ab);
  std::vector<std::pair<double, int>> near;
  std::vector<Vec3> witnesses(static_cast<size_t>(points.rows()));
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    const Vec3 p = points.row(i).transpose();
    const double t = denom < 1e-12 ? 0.0
                                   : std::min(std::max((p - p0).dot(ab) / denom, 0.0), 1.0);
    const Vec3 witness = p0 + t * ab;
    witnesses[static_cast<size_t>(i)] = witness;
    const double gap = (p - witness).norm() - radius;
    if (gap < influence) {
      near.emplace_back(gap, static_cast<int>(i));
    }
  }
  if (near.empty()) {
    return {};
  }
  const size_t keep = std::min<size_t>(kPairsPerLink, near.size());
  std::partial_sort(near.begin(), near.begin() + keep, near.end());
  std::vector<NearPoint> out;
  out.reserve(keep);
  for (size_t k = 0; k < keep; ++k) {
    const int i = near[k].second;
    out.push_back(NearPoint{i, witnesses[static_cast<size_t>(i)],
                            points.row(i).transpose(), near[k].first});
  }
  return out;
}

std::vector<MotionLayer::NearLine> MotionLayer::near_lines(const Points &voxels, const Vec3 &eye,
                                                          const Vec3 &p0, const Vec3 &p1,
                                                          double radius,
                                                          double influence) const {
  if (voxels.rows() == 0) {
    return {};
  }
  const Vec3 centre = (p0 + p1) / 2.0;
  const Vec3 to_centre = centre - eye;
  const double reach = to_centre.norm();
  const double half = (p1 - p0).norm() / 2.0 + radius + influence;
  const double cos_limit = std::cos(std::atan2(half, std::max(reach, 1e-6)));

  std::vector<int> idx;
  for (Eigen::Index i = 0; i < voxels.rows(); ++i) {
    const Vec3 ray = voxels.row(i).transpose() - eye;
    const double length = ray.norm();
    // the voxel is beyond the capsule, not in front of it
    if (length <= reach - half) {
      continue;
    }
    if (ray.dot(to_centre) < cos_limit * length * reach) {
      continue;
    }
    idx.push_back(static_cast<int>(i));
  }
  if (idx.empty()) {
    return {};
  }

  std::vector<Vec3> samples(8);           // 8 points along the capsule
  for (int s = 0; s < 8; ++s) {
    samples[s] = p0 + (s / 7.0) * (p1 - p0);
  }
  std::vector<std::pair<double, size_t>> best;
  std::vector<Vec3> best_witness(idx.size());
  std::vector<Vec3> best_on_line(idx.size());
  for (size_t k = 0; k < idx.size(); ++k) {
    const Vec3 d_line = voxels.row(idx[k]).transpose() - eye;
    const double denom = std::max(d_line.dot(d_line), 1e-12);
    double closest = kInf;
    for (int s = 0; s < 8; ++s) {
      const double tpar =
          std::min(std::max((samples[s] - eye).dot(d_line) / denom, 0.0), 1.0);
      const Vec3 on_line = eye + tpar * d_line;
      const double dist = (samples[s] - on_line).norm() - radius;
      if (dist < closest) {
        closest = dist;
        best_witness[k] = samples[s];
        best_on_line[k] = on_line;
      }
    }
    if (closest < influence) {
      best.emplace_back(closest, k);
    }
  }
  if (best.empty()) {
    return {};
  }
  const size_t keep = std::min<size_t>(kPairsPerLink, best.size());
  std::partial_sort(best.begin(), best.begin() + keep, best.end());
  std::vector<NearLine> out;
  out.reserve(keep);
  for (size_t n = 0; n < keep; ++n) {
    const size_t k = best[n].second;
    out.push_back(NearLine{best_witness[k], best_on_line[k], best[n].first});
  }
  return out;
}

}  // namespace sightline
