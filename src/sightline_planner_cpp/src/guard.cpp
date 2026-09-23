#include "sightline_planner/guard.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <set>

#include <Eigen/QR>

namespace sightline {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

TrajectoryGuard::TrajectoryGuard(ToolKinematics *kin, ViewGrid *grid,
                                 const std::vector<Plane> &planes)
    : kin_(kin), grid_(grid) {
  for (const Plane &p : planes) {
    planes_.push_back(Plane{p.point, p.normal / p.normal.norm()});
  }
  blind_pad_ = 0.5 * std::sqrt(3.0) * kBlindCellM;

  const mjModel *m = kin_->model();
  std::vector<double> radii;
  std::vector<int> bodies;
  // each joint, by the body that turns on it, with that body's thickest shape
  for (const char *name : {"ur5e/shoulder_link", "ur5e/upper_arm_link", "ur5e/forearm_link",
                           "ur5e/wrist_1_link", "ur5e/wrist_2_link", "ur5e/wrist_3_link"}) {
    joint_bodies_.push_back(mj_name2id(m, mjOBJ_BODY, name));
  }
  for (int b : joint_bodies_) {
    double own = -kInf;
    for (int g = 0; g < m->ngeom; ++g) {
      if (m->geom_group[g] == 3 && m->geom_bodyid[g] == b) {
        own = std::max(own, shape_line(m, g).second);
      }
    }
    radii.push_back(std::isfinite(own) ? own : 0.06);
    bodies.push_back(b);
  }
  tip_ = mj_name2id(m, mjOBJ_SITE, "tool_tip");
  const int wrist = mj_name2id(m, mjOBJ_BODY, "ur5e/wrist_3_link");
  radii.push_back(0.0085);     // the bit, whose tip this is
  radii.push_back(0.032);      // the wrist camera's box corner
  bodies.push_back(wrist);
  bodies.push_back(wrist);
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_group[g] != 3) {
      continue;
    }
    const auto [half, radius] = shape_line(m, g);
    const int n = half > 0 ? static_cast<int>(std::ceil(2 * half / kSampleM)) + 1 : 1;
    const double spacing = n > 1 ? 2 * half / (n - 1) : 0.0;
    for (int k = 0; k < n; ++k) {
      const double t = n > 1 ? -1.0 + 2.0 * k / (n - 1) : 0.0;
      sample_geom_.push_back(g);
      sample_off_.push_back(t * half);
      radii.push_back(radius + 0.5 * spacing);
      bodies.push_back(m->geom_bodyid[g]);
    }
  }
  radii_ = VecX(static_cast<Eigen::Index>(radii.size()));
  for (size_t i = 0; i < radii.size(); ++i) {
    radii_[static_cast<Eigen::Index>(i)] = radii[i];
  }
  cam_body_ = m->cam_bodyid[kin_->camera_id()];
  cam_local_ = Vec3(m->cam_pos[3 * kin_->camera_id()], m->cam_pos[3 * kin_->camera_id() + 1],
                    m->cam_pos[3 * kin_->camera_id() + 2]);
  // the shoulder is bolted on at worktop height and only turns about the vertical,
  // so the worktop is not a limit for it
  above_worktop_.resize(bodies.size());
  for (size_t i = 0; i < bodies.size(); ++i) {
    above_worktop_[i] = bodies[i] != joint_bodies_[0] ? 1 : 0;
  }

  // Points no joint can move, the shoulder on its own axis, are left out of every
  // decision: nothing the arm does changes them. He can still walk up to the base,
  // and the judge will say who moved in. Left in, they froze the arm whenever he
  // put a part into the jig, which sits 18 cm from the shoulder.
  //
  // The eight sample poses come from a fixed seed, as they did in the first version
  // of this file, so the answer is the same on every run. The two tests below are
  // coarse enough that which eight poses they are does not matter.
  std::mt19937_64 rng(0);
  std::uniform_real_distribution<double> spin(-M_PI, M_PI);
  const size_t n_points = radii.size();
  std::vector<double> spread(n_points, 0.0);
  std::vector<double> off_axis(n_points, 0.0);
  const Points ref = points(Vec6::Zero());
  const Vec3 axis = ref.row(0).transpose();       // the shoulder joint sits on the base's own axis
  for (int trial = 0; trial < 8; ++trial) {
    Vec6 q;
    for (int j = 0; j < kNq; ++j) {
      q[j] = spin(rng);
    }
    const Points pts = points(q);
    for (size_t i = 0; i < n_points; ++i) {
      const Vec3 p = pts.row(static_cast<Eigen::Index>(i)).transpose();
      spread[i] = std::max(spread[i], (p - ref.row(static_cast<Eigen::Index>(i)).transpose()).norm());
      off_axis[i] = std::max(off_axis[i], (p.head<2>() - axis.head<2>()).norm());
    }
  }
  moves_.resize(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    moves_[i] = spread[i] > 1e-3 ? 1 : 0;
    if (moves_[i]) {
      moves_idx_.push_back(static_cast<int>(i));
    }
  }
  radii_moves_ = VecX(static_cast<Eigen::Index>(moves_idx_.size()));
  above_worktop_moves_.resize(moves_idx_.size());
  rooted_.resize(moves_idx_.size());
  for (size_t k = 0; k < moves_idx_.size(); ++k) {
    const size_t i = static_cast<size_t>(moves_idx_[k]);
    radii_moves_[static_cast<Eigen::Index>(k)] = radii_[static_cast<Eigen::Index>(i)];
    above_worktop_moves_[k] = above_worktop_[i];
    // Points that never leave the base's axis by more than 0.2 m, the root of the
    // upper arm, cannot get out of anyone's way. They must never move closer to him,
    // but they do not send the arm fleeing: while he put a part in the jig 20 cm from
    // the base, they kept the whole arm escaping for 1.5 s, all the way to park.
    rooted_[k] = off_axis[i] < 0.20 ? 1 : 0;
  }
}

Points TrajectoryGuard::points(const Vec6 &q) const {
  const mjModel *m = kin_->model();
  mjData *d = kin_->data();
  for (int j = 0; j < kNq; ++j) {
    d->qpos[j] = q[j];
  }
  mj_kinematics(m, d);
  const Eigen::Index n =
      static_cast<Eigen::Index>(joint_bodies_.size() + 2 + sample_geom_.size());
  Points out(n, 3);
  Eigen::Index row = 0;
  for (int b : joint_bodies_) {
    out.row(row++) = Vec3(d->xpos[3 * b], d->xpos[3 * b + 1], d->xpos[3 * b + 2]).transpose();
  }
  out.row(row++) = Vec3(d->site_xpos[3 * tip_], d->site_xpos[3 * tip_ + 1],
                        d->site_xpos[3 * tip_ + 2]).transpose();
  Mat3 cam_R;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      cam_R(r, c) = d->xmat[9 * cam_body_ + 3 * r + c];
    }
  }
  const Vec3 cam_origin(d->xpos[3 * cam_body_], d->xpos[3 * cam_body_ + 1],
                        d->xpos[3 * cam_body_ + 2]);
  out.row(row++) = (cam_origin + cam_R * cam_local_).transpose();
  for (size_t k = 0; k < sample_geom_.size(); ++k) {
    const int g = sample_geom_[k];
    const Vec3 centre(d->geom_xpos[3 * g], d->geom_xpos[3 * g + 1], d->geom_xpos[3 * g + 2]);
    const Vec3 axis(d->geom_xmat[9 * g + 2], d->geom_xmat[9 * g + 5], d->geom_xmat[9 * g + 8]);
    out.row(row++) = (centre + axis * sample_off_[k]).transpose();
  }
  return out;
}

Grid TrajectoryGuard::shadow(const Vec6 &q) const {
  return grid_->shadow_of(points(q), radii_);
}

VecX TrajectoryGuard::excess_of(const Points &all_points, double at) const {
  checks_ += 1;
  const Eigen::Index n = static_cast<Eigen::Index>(moves_idx_.size());
  Points pts(n, 3);
  for (Eigen::Index k = 0; k < n; ++k) {
    pts.row(k) = all_points.row(moves_idx_[static_cast<size_t>(k)]);
  }
  VecX e = grid_->excess(pts, radii_moves_, at).depth;
  if (blind_.rows() > 0) {
    for (Eigen::Index k = 0; k < n; ++k) {
      const double reach = grid_->safe() + radii_moves_[k] + blind_pad_;
      double gap = kInf;
      for (Eigen::Index j = 0; j < blind_.rows(); ++j) {
        gap = std::min(gap, (pts.row(k) - blind_.row(j)).norm());
      }
      e[k] = std::max(e[k], reach - gap);
    }
  }
  for (const Plane &plane : planes_) {
    for (Eigen::Index k = 0; k < n; ++k) {
      if (!above_worktop_moves_[static_cast<size_t>(k)]) {
        continue;
      }
      const double under =
          radii_moves_[k] - (pts.row(k).transpose() - plane.point).dot(plane.normal);
      e[k] = std::max(e[k], under);
    }
  }
  return e;
}

std::vector<char> TrajectoryGuard::bad(const Points &pts, double at) const {
  const VecX e = excess_of(pts, at);
  std::vector<char> out(static_cast<size_t>(e.size()));
  for (Eigen::Index i = 0; i < e.size(); ++i) {
    out[static_cast<size_t>(i)] = e[i] > 0 ? 1 : 0;
  }
  return out;
}

GuardExplain TrajectoryGuard::explain(const Vec6 &q, double at) const {
  const Points all_points = points(q);
  const Eigen::Index n = static_cast<Eigen::Index>(moves_idx_.size());
  Points pts(n, 3);
  for (Eigen::Index k = 0; k < n; ++k) {
    pts.row(k) = all_points.row(moves_idx_[static_cast<size_t>(k)]);
  }
  const Excess e = grid_->excess(pts, radii_moves_, at, true);
  GuardExplain out;
  double deepest = 0.0;
  Eigen::Index worst = -1;
  for (Eigen::Index i = 0; i < e.depth.size(); ++i) {
    if (e.depth[i] > 0) {
      out.grid += 1;
    }
    if (worst < 0 || e.depth[i] > e.depth[worst]) {
      worst = i;
    }
    deepest = std::max(deepest, e.depth[i]);
  }
  out.deepest_mm = std::round(std::max(deepest, 0.0) * 1000.0 * 10.0) / 10.0;
  if (worst >= 0 && e.depth[worst] > 0) {
    const int k = e.cells[static_cast<size_t>(worst)];
    out.has_worst = true;
    out.worst_point = pts.row(worst).transpose();
    out.his_cell = grid_->cell_point(k);
    out.cell_hidden = grid_->hidden()[static_cast<size_t>(k)] != 0;
    out.cell_speed = grid_->cell_speed()[static_cast<size_t>(k)];
    out.cell_age = at - grid_->seen_at()[static_cast<size_t>(k)];
  }
  if (blind_.rows() > 0) {
    for (Eigen::Index k = 0; k < n; ++k) {
      double gap = kInf;
      for (Eigen::Index j = 0; j < blind_.rows(); ++j) {
        gap = std::min(gap, (pts.row(k) - blind_.row(j)).norm());
      }
      if (gap < grid_->safe() + radii_moves_[k] + blind_pad_) {
        out.blind += 1;
      }
    }
  }
  for (const Plane &plane : planes_) {
    for (Eigen::Index k = 0; k < n; ++k) {
      if (above_worktop_moves_[static_cast<size_t>(k)] &&
          (pts.row(k).transpose() - plane.point).dot(plane.normal) < radii_moves_[k]) {
        out.worktop += 1;
      }
    }
  }
  return out;
}

int TrajectoryGuard::count(const Vec6 &q, double at) const {
  const std::vector<char> flags = bad(points(q), at);
  return static_cast<int>(std::count(flags.begin(), flags.end(), 1));
}

bool TrajectoryGuard::pose_clear(const Vec6 &q, double at) const {
  const std::vector<char> flags = bad(points(q), at);
  return std::find(flags.begin(), flags.end(), 1) == flags.end();
}

// The root of the upper arm stays near the person whatever the arm does while they
// work at the jig, so a plan must not wait for it to clear.
bool TrajectoryGuard::pose_ok(const Vec6 &q, double at, const Vec6 &q_from) const {
  const Points pts = points(q);
  const VecX e = excess_of(pts, at);
  bool any_inside = false;
  for (Eigen::Index i = 0; i < e.size(); ++i) {
    if (e[i] > 0) {
      any_inside = true;
      break;
    }
  }
  if (!any_inside) {
    return true;
  }
  const Points start = points(q_from);
  const VecX ref = excess_of(start, at);
  for (Eigen::Index k = 0; k < e.size(); ++k) {
    if (e[k] <= 0) {
      continue;
    }
    const int i = moves_idx_[static_cast<size_t>(k)];
    const bool stays = (pts.row(i) - start.row(i)).norm() < kStaysM;
    if (!(stays && e[k] <= std::max(ref[k], 0.0) + kTolM)) {
      return false;
    }
  }
  return true;
}

std::pair<bool, double> TrajectoryGuard::path_clear(const std::vector<Vec6> &configs,
                                                    const std::vector<double> &times) const {
  for (size_t i = 0; i < configs.size() && i < times.size(); ++i) {
    const std::vector<char> flags = bad(points(configs[i]), times[i]);
    if (std::find(flags.begin(), flags.end(), 1) != flags.end()) {
      return {false, times[i]};
    }
  }
  return {true, kInf};
}

void TrajectoryGuard::stop_after(const Vec6 &q, const Vec6 &qd, double now,
                                 std::vector<Vec6> *configs, std::vector<double> *times,
                                 double keep_s) {
  configs->clear();
  times->clear();
  const double fastest = qd.cwiseAbs().maxCoeff();
  configs->push_back(q + qd * keep_s);
  times->push_back(now + keep_s);
  if (fastest > 1e-9) {
    const double t_stop = fastest / kStopDecel;
    const int steps = std::max(1, static_cast<int>(std::ceil(t_stop / kStepS)));
    for (int k = 1; k <= steps; ++k) {
      const double s = std::min(k * kStepS, t_stop);
      // distance covered while slowing linearly to zero
      const double frac = s - 0.5 * s * s / t_stop;
      configs->push_back(q + qd * keep_s + qd * frac);
      times->push_back(now + keep_s + s);
    }
  }
}

void TrajectoryGuard::line(const Vec6 &q, const Vec6 &goal, double speed, double now,
                           std::vector<Vec6> *configs, std::vector<double> *times, double step) {
  configs->clear();
  times->clear();
  const double span = (goal - q).cwiseAbs().maxCoeff();
  if (span < 1e-9) {
    configs->push_back(q);
    times->push_back(now);
    return;
  }
  const int n = std::max(1, static_cast<int>(std::ceil(span / step)));
  for (int k = 1; k <= n; ++k) {
    configs->push_back(q + (goal - q) * (static_cast<double>(k) / n));
    times->push_back(now + span * (static_cast<double>(k) / n) / speed);
  }
}

Vec6 TrajectoryGuard::toward(const Vec6 &qd_prev, const Vec6 &target, double rate) {
  const double step = rate * kCycleS;
  Vec6 out;
  for (int j = 0; j < kNq; ++j) {
    out[j] = qd_prev[j] + std::min(std::max(target[j] - qd_prev[j], -step), step);
  }
  return out;
}

// go is the task's own command, slow is its direction at half or a quarter of the
// speed, brake slows everything to a stop, and escape heads for whichever of the way
// it came, the stand-off or park takes the most depth off the arm.
std::pair<Vec6, std::string> TrajectoryGuard::command(const Vec6 &q, const Vec6 &qd_prev,
                                                      const Vec6 &qd_want, double now,
                                                      const std::vector<EscapeTarget> &targets) {
  const auto t0 = std::chrono::steady_clock::now();
  const Points here = points(q);
  std::map<long long, VecX> held;
  // How deep each point would be at that time if the arm stood where it is.
  auto hold = [&](double at) -> const VecX & {
    const long long key = std::llround(at * 1e6);
    auto it = held.find(key);
    if (it == held.end()) {
      it = held.emplace(key, excess_of(here, at)).first;
    }
    return it->second;
  };

  const VecX inside = hold(now + kCycleS);
  bool any_inside = false;
  bool any_inside_free = false;      // ignoring the points no joint can move away
  for (Eigen::Index i = 0; i < inside.size(); ++i) {
    if (inside[i] > 0) {
      any_inside = true;
      if (!rooted_[static_cast<size_t>(i)]) {
        any_inside_free = true;
      }
    }
  }
  if (!any_inside) {
    history_.emplace_back(now, q);
  }
  while (!history_.empty() && history_.front().first < now - kHistoryS) {
    history_.pop_front();
  }

  std::string verdict;
  Vec6 qd = qd_want;
  bool have_best = false;
  EscapeChoice best;
  if (any_inside_free) {
    best = escape(q, qd_prev, now, targets);
    have_best = true;
    const VecX &later = hold(now + kEscapeSamples[1]);
    double staying = 0.0;
    for (Eigen::Index i = 0; i < later.size(); ++i) {
      if (!rooted_[static_cast<size_t>(i)]) {
        staying += std::max(later[i], 0.0);
      }
    }
    if (best.last_score < staying - kEscapeGainM) {
      verdict = best.name;
      qd = best.first;
    }
  }

  if (verdict.empty()) {
    std::vector<std::pair<std::string, Vec6>> tries;
    tries.emplace_back("go", qd_want);
    if (!any_inside_free) {      // inside the margin a slower way in is no better
      tries.emplace_back("slow", toward(qd_prev, 0.5 * qd_want));
      tries.emplace_back("slow", toward(qd_prev, 0.25 * qd_want));
    }
    tries.emplace_back("brake", toward(qd_prev, Vec6::Zero()));
    for (const auto &[name, candidate] : tries) {
      std::vector<Vec6> configs;
      std::vector<double> times;
      stop_after(q, candidate, now, &configs, &times);
      // No point deeper than holding still would leave it at the same moment; a clear
      // point must stay clear.
      bool no_worse = true;
      for (size_t i = 0; i < configs.size() && no_worse; ++i) {
        const VecX &ref = hold(times[i]);
        const VecX e = excess_of(points(configs[i]), times[i]);
        for (Eigen::Index k = 0; k < e.size(); ++k) {
          const double allowed = ref[k] > 0 ? ref[k] + kTolM : 0.0;
          if (e[k] > allowed) {
            no_worse = false;
            break;
          }
        }
      }
      if (no_worse) {
        verdict = name;
        qd = candidate;
        break;
      }
    }
  }

  if (verdict.empty()) {
    if (!have_best) {
      best = escape(q, qd_prev, now, targets);
    }
    verdict = best.name;
    qd = best.first;
  }

  verdicts_[verdict] += 1;
  last_ = verdict;
  ms_.push_back(
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  return {qd, verdict};
}

TrajectoryGuard::EscapeChoice TrajectoryGuard::escape(
    const Vec6 &q, const Vec6 &qd_prev, double now,
    const std::vector<EscapeTarget> &targets) const {
  struct Goal {
    std::string name;
    Vec6 pose = Vec6::Zero();
    bool has_pose = false;
    bool up = false;
  };
  std::vector<Goal> goals;
  Vec6 back;
  if (retrace(q, now, &back)) {
    goals.push_back(Goal{"escape back", back, true, false});
  }
  for (const EscapeTarget &target : targets) {
    if (!target.has_pose) {
      continue;
    }
    // the same pose twice, or where the arm already is, is not another way out
    if ((target.pose - q).cwiseAbs().maxCoeff() <= 0.03) {
      continue;
    }
    bool distinct = true;
    for (const Goal &other : goals) {
      if (other.has_pose && (target.pose - other.pose).cwiseAbs().maxCoeff() <= 1e-6) {
        distinct = false;
        break;
      }
    }
    if (distinct) {
      goals.push_back(Goal{"escape to " + target.name, target.pose, true, false});
    }
  }
  // straight up, toward the camera: away from anything hidden under the arm, and
  // the one way out when the arm is already where the other ways lead
  goals.push_back(Goal{"escape up", Vec6::Zero(), false, true});

  std::vector<Goal> all;
  all.push_back(Goal{"escape brake", Vec6::Zero(), false, false});
  all.insert(all.end(), goals.begin(), goals.end());

  EscapeChoice best;
  bool have = false;
  for (const Goal &goal : all) {
    Vec6 first = Vec6::Zero();
    std::vector<double> scores;
    rollout(q, qd_prev, goal.has_pose ? &goal.pose : nullptr, goal.up, now, &first, &scores);
    const double last_score = scores.empty() ? 0.0 : scores.back();
    double total = 0.0;
    for (double s : scores) {
      total += s;
    }
    if (!have || last_score < best.last_score ||
        (last_score == best.last_score && total < best.total_score)) {
      best = EscapeChoice{last_score, total, first, goal.name};
      have = true;
    }
  }
  return best;
}

void TrajectoryGuard::rollout(const Vec6 &q, const Vec6 &qd_prev, const Vec6 *goal, bool up,
                              double now, Vec6 *first, std::vector<double> *scores) const {
  Vec6 qk = q;
  Vec6 qdk = qd_prev;
  bool have_first = false;
  scores->clear();
  std::set<int> marks;
  for (double s : kEscapeSamples) {
    marks.insert(static_cast<int>(std::lround(s / kCycleS)));
  }
  Vec6 lift = Vec6::Zero();
  bool have_lift = false;
  const int steps = static_cast<int>(std::lround(kEscapeSamples[1] / kCycleS));
  for (int k = 1; k <= steps; ++k) {
    Vec6 target = Vec6::Zero();
    if (up) {
      if (!have_lift || k % 5 == 0) {
        Eigen::Matrix<double, 6, 1> twist;
        twist << 0.0, 0.0, kUpSpeed, 0.0, 0.0, 0.0;
        lift = kin_->jacobian(qk).completeOrthogonalDecomposition().pseudoInverse() * twist;
        const double fastest = lift.cwiseAbs().maxCoeff();
        if (fastest > kEscapeSpeed) {
          lift *= kEscapeSpeed / fastest;
        }
        have_lift = true;
      }
      target = lift;
    } else if (goal != nullptr) {
      const Vec6 left = *goal - qk;
      const double dist = left.cwiseAbs().maxCoeff();
      const double speed =
          std::min({kEscapeSpeed, std::sqrt(2.0 * kStopDecel * dist), dist / kCycleS});
      target = left / std::max(dist, 1e-9) * speed;
    }
    qdk = toward(qdk, target);
    qk = qk + qdk * kCycleS;
    if (!have_first) {
      *first = qdk;
      have_first = true;
    }
    if (marks.count(k) > 0) {
      const VecX e = excess_of(points(qk), now + k * kCycleS);
      double depth = 0.0;
      for (Eigen::Index i = 0; i < e.size(); ++i) {
        if (!rooted_[static_cast<size_t>(i)]) {
          depth += std::max(e[i], 0.0);
        }
      }
      scores->push_back(depth);
    }
  }
}

bool TrajectoryGuard::retrace(const Vec6 &q, double now, Vec6 *out) const {
  int tried = 0;
  for (int i = static_cast<int>(history_.size()) - 1; i >= 0; i -= 5) {
    const Vec6 &past = history_[static_cast<size_t>(i)].second;
    if ((past - q).cwiseAbs().maxCoeff() < 0.03) {
      continue;
    }
    tried += 1;
    if (pose_clear(past, now)) {
      *out = past;
      return true;
    }
    if (tried >= 2) {
      break;
    }
  }
  return false;
}

}  // namespace sightline
