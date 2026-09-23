#include "sightline_planner/b0.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sightline {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double round_to(double value, int places) {
  const double scale = std::pow(10.0, places);
  return std::round(value * scale) / scale;
}

}  // namespace

Mat3 rot_z(double a) {
  const double c = std::cos(a);
  const double s = std::sin(a);
  Mat3 out;
  out << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  return out;
}

const Mat3 &bit_down() {
  static const Mat3 value = [] {
    Mat3 m;
    m << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
    return m;
  }();
  return value;
}

const std::vector<std::string> &B0::stages() {
  static const std::vector<std::string> value{"empty", "rail_ready", "cover_on", "cover_ready",
                                              "done"};
  return value;
}

int B0::stage_index(const std::string &signal) {
  const auto &all = stages();
  return static_cast<int>(std::find(all.begin(), all.end(), signal) - all.begin());
}

B0::B0(ToolKinematics *kin, const Taught &taught, const Vec3 &park_tip)
    : kin_(kin), taught_(taught), park_tip_(park_tip) {}

void B0::build(const Vec6 *home_q) {
  screws_.clear();
  for (size_t i = 0; i < taught_.rail_holes.size(); ++i) {
    screws_.push_back(Screw{"rail_" + std::to_string(i + 1), taught_.world(taught_.rail_holes[i]),
                            "rail_ready", 0.0, false});
  }
  for (size_t i = 0; i < taught_.cover_holes.size(); ++i) {
    screws_.push_back(Screw{"cover_" + std::to_string(i + 1),
                            taught_.world(taught_.cover_holes[i]), "cover_ready", 0.0, false});
  }
  for (Screw &s : screws_) {
    s.turn_deg = pick_turn(s.hole);
  }
  Vec6 home;
  home << 0.0, -1.6, 1.6, -1.6, -1.57, 0.0;
  teach(home_q == nullptr ? home : *home_q);
}

void B0::teach(const Vec6 &home) {
  // Solve every pose the cycle visits once, as one chain from home, and keep them.
  //
  // A programmer teaches a cell this way: each pose a short joint move from the
  // last, all in one arm configuration. Solving each target afresh, nearest to
  // wherever the arm happens to be, let a 15 mm drift while backing away from the
  // worker flip the solver onto another branch, and the next move dived 40 cm under
  // its own path to get there.
  poses_.clear();
  Vec6 q = home;
  double err = kInf;
  const Vec6 park = kin_->ik_multi(park_tip_, &bit_down(), q, &err);
  if (std::isfinite(err)) {
    poses_["park:park"] = park;
    q = park;
  }
  for (const Screw &screw : screws_) {
    const Mat3 R = rot_z(screw.turn_deg * M_PI / 180.0) * bit_down();
    const Vec3 feeder = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.approach);
    const Vec6 q_feed = kin_->ik_multi(feeder, &R, q, &err);
    if (std::isfinite(err)) {
      poses_["to_feeder:" + screw.name] = q_feed;
      q = q_feed;
    }
    Vec3 hole;
    Mat3 unused;
    pose_for(screw, taught_.screw_length + taught_.approach, &hole, &unused);
    const Vec6 q_hole = kin_->ik_multi(hole, &R, q, &err);
    if (std::isfinite(err)) {
      poses_["to_hole:" + screw.name] = q_hole;
      // where to wait for this screw if he is in the way: back toward the robot
      // and up, a short move from the hole, out of his space and behind the jig
      // as the camera sees it
      const Vec3 back = screw.hole + Vec3(0.0, kStandoffBackM, kStandoffUpM);
      const Vec6 q_wait = kin_->ik_multi(back, &R, q_hole, &err);
      if (std::isfinite(err)) {
        poses_["standoff:" + screw.name] = q_wait;
      }
      q = q_hole;
    }
  }
}

// The screwdriver mounts from the side, so the turn angle moves the whole arm and
// some angles have no solution at all (gate 1). Taking the first angle that solves is
// not enough: at one of them the arm sits next to a configuration change and swings
// 37 mm sideways on the way in. Each angle is scored by how far the joints move
// between the approach pose and the hole, and the quietest one wins.
double B0::pick_turn(const Vec3 &hole) const {
  const Vec3 above = hole + Vec3(0.0, 0.0, taught_.screw_length + taught_.approach);
  const Vec3 down = hole + Vec3(0.0, 0.0, taught_.screw_length);
  Vec6 home;
  home << 0.0, -1.6, 1.6, -1.6, -1.57, 0.0;
  double best = 0.0;
  double best_score = kInf;
  for (int psi = 0; psi < 360; psi += kTurnStepDeg) {
    const Mat3 R = rot_z(psi * M_PI / 180.0) * bit_down();
    double err = kInf;
    const Vec6 q_up = kin_->ik_multi(above, &R, home, &err);
    if (!std::isfinite(err)) {
      continue;
    }
    const Vec6 q_dn = kin_->ik(down, &R, q_up, &err, 200);
    if (err > 1e-4) {
      continue;
    }
    const double score = (q_dn - q_up).cwiseAbs().maxCoeff();
    if (score < best_score) {
      best = psi;
      best_score = score;
    }
  }
  return best;
}

void B0::pose_for(const Screw &screw, double height, Vec3 *pos, Mat3 *R) const {
  *pos = screw.hole + Vec3(0.0, 0.0, height);
  *R = rot_z(screw.turn_deg * M_PI / 180.0) * bit_down();
}

std::optional<Vec3> B0::find_hole(const SensorFrame &frame, const Screw &screw) {
  const Image &img = frame.wrist_rgb;
  if (img.empty()) {
    return std::nullopt;
  }
  const auto [cam_pos, cam_R] = kin_->camera_pose(frame.q);
  const Vec3 rel = cam_R.transpose() * (screw.hole - cam_pos);
  if (rel[2] >= -1e-6) {                        // the camera looks along its own -z
    return std::nullopt;
  }
  const int h = img.height;
  const int w = img.width;
  const double f = (h / 2.0) / std::tan(kin_->camera_fovy() * M_PI / 180.0 / 2.0);
  const double u = w / 2.0 + f * (rel[0] / -rel[2]);
  const double v = h / 2.0 - f * (rel[1] / -rel[2]);
  const int r = 45;
  const int u0 = static_cast<int>(std::max(u - r, 0.0));
  const int u1 = static_cast<int>(std::min(u + r, static_cast<double>(w)));
  const int v0 = static_cast<int>(std::max(v - r, 0.0));
  const int v1 = static_cast<int>(std::min(v + r, static_cast<double>(h)));
  if (u1 - u0 < 10 || v1 - v0 < 10) {
    hole_missed_ += 1;
    return std::nullopt;
  }
  std::vector<double> win;
  win.reserve(static_cast<size_t>(u1 - u0) * (v1 - v0));
  for (int vv = v0; vv < v1; ++vv) {
    for (int uu = u0; uu < u1; ++uu) {
      double sum = 0.0;
      for (int c = 0; c < img.channels; ++c) {
        sum += img.at(vv, uu, c);
      }
      win.push_back(sum / std::max(img.channels, 1));
    }
  }
  std::vector<double> sorted = win;
  std::sort(sorted.begin(), sorted.end());
  const size_t mid = sorted.size() / 2;
  const double median =
      sorted.size() % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
  const double cut = 0.55 * median;
  double sum_u = 0.0;
  double sum_v = 0.0;
  int n = 0;
  const int ww = u1 - u0;
  for (size_t i = 0; i < win.size(); ++i) {
    if (win[i] < cut) {
      sum_u += static_cast<double>(i % ww);
      sum_v += static_cast<double>(i / ww);
      n += 1;
    }
  }
  if (n < 12 || n > 0.5 * static_cast<double>(win.size())) {  // too small to be a hole, or the view is dark
    hole_missed_ += 1;
    return std::nullopt;
  }
  const double cu = u0 + sum_u / n;
  const double cv = v0 + sum_v / n;
  // back along the ray to the plane the hole sits in
  const Vec3 ray = cam_R * Vec3((cu - w / 2.0) / f, -(cv - h / 2.0) / f, -1.0);
  if (std::abs(ray[2]) < 1e-6) {
    hole_missed_ += 1;
    return std::nullopt;
  }
  const double s = (screw.hole[2] - cam_pos[2]) / ray[2];
  hole_seen_ += 1;
  return cam_pos + s * ray;
}

Vec6 B0::update(const SensorFrame &frame, const Obstacles *obstacles) {
  const Vec6 qd = task(frame);
  if (motion_ == nullptr || obstacles == nullptr) {
    return qd;
  }
  const Report report = motion_->solve_joint(frame.q, qd, *obstacles);
  reports_.push_back(report);
  return report.qd;
}

Vec6 B0::task(const SensorFrame &frame) {
  now_ = frame.t;
  const Vec6 qd = task_inner(frame);
  last_t_ = frame.t;
  return qd;
}

void B0::drop_passed(const SensorFrame &frame) {
  while (index_ < static_cast<int>(screws_.size())) {
    Screw &screw = screws_[static_cast<size_t>(index_)];
    const bool at_the_hole = state_ == "descend" || state_ == "drive";
    if (stage_index(frame.jig_signal) > stage_index(screw.needs) && !screw.driven && !at_the_hole) {
      const std::string why = frame.jig_signal == "done"
                                  ? "the box was taken before it was driven"
                                  : "the next part went on before it was driven";
      missed_.push_back(MissedEntry{round_to(frame.t, 2), screw.name, why});
      index_ += 1;
      state_ = "wait";
      resume_state_.reset();
      continue;
    }
    break;
  }
}

Vec6 B0::task_inner(const SensorFrame &frame) { return base_task_inner(frame); }

Vec6 B0::base_task_inner(const SensorFrame &frame) {
  const Vec6 q = frame.q;
  drop_passed(frame);
  if (index_ >= static_cast<int>(screws_.size())) {
    state_ = "park";
  }
  Screw *screw = index_ < static_cast<int>(screws_.size()) ? &screws_[static_cast<size_t>(index_)]
                                                           : nullptr;
  double left = 0.0;
  double err = 0.0;

  if (state_ == "wait") {
    if (screw != nullptr && frame.jig_signal == screw->needs) {
      go("to_feeder", frame.t);
    }
    return Vec6::Zero();
  }

  if (state_ == "to_feeder") {
    const Vec3 target = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.approach);
    const Mat3 R = rot_z(screw->turn_deg * M_PI / 180.0) * bit_down();
    const Vec6 qd = goto_pose(q, target, R, key_for("to_feeder"), &left);
    if (left < 0.004) {
      go("pick", frame.t, kPickS);
    }
    return qd;
  }

  if (state_ == "pick") {
    const Vec3 target = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.screw_length);
    const Mat3 R = rot_z(screw->turn_deg * M_PI / 180.0) * bit_down();
    const Vec6 qd = move_to(q, target, R, &err, 0.10);
    if (frame.t >= until_ && err < 0.004) {
      go("to_hole", frame.t);
    }
    return qd;
  }

  if (state_ == "to_hole") {
    Vec3 pos;
    Mat3 R;
    pose_for(*screw, taught_.screw_length + taught_.approach, &pos, &R);
    const Vec6 qd = goto_pose(q, pos, R, key_for("to_hole"), &left);
    if (left < 0.004) {
      corrected_.reset();
      go("servo", frame.t, 0.4);
    }
    return qd;
  }

  if (state_ == "servo") {
    const std::optional<Vec3> seen = find_hole(frame, *screw);
    if (seen.has_value()) {
      const double off = (seen->head<2>() - screw->hole.head<2>()).norm();
      hole_error_mm_.push_back(round_to(off * 1000, 2));
      if (off <= kMaxCorrectionM) {
        corrected_ = seen;
      } else {
        hole_rejected_ += 1;
      }
    }
    Vec3 pos;
    Mat3 R;
    pose_for(*screw, taught_.screw_length + taught_.approach, &pos, &R);
    if (corrected_.has_value()) {
      pos = Vec3((*corrected_)[0], (*corrected_)[1], pos[2]);
    }
    const Vec6 qd = move_to(q, pos, R, &err);
    if (frame.t >= until_ && err < 0.002) {
      go("descend", frame.t);
    }
    return qd;
  }

  if (state_ == "descend") {
    Vec3 pos;
    Mat3 R;
    pose_for(*screw, taught_.screw_length, &pos, &R);
    if (corrected_.has_value()) {
      pos = Vec3((*corrected_)[0], (*corrected_)[1], pos[2]);
    }
    const Vec3 above(pos[0], pos[1], pos[2] + taught_.approach);
    const Vec6 qd =
        straight(q, above, pos, R, key_for("descend"), 25.0 * M_PI / 180.0, &left);
    if (left < 0.003) {
      go("drive", frame.t, kDriveS);
    }
    return qd;
  }

  if (state_ == "drive") {
    if (frame.t >= until_) {
      screw->driven = true;
      go("retract", frame.t);
    }
    return Vec6::Zero();
  }

  if (state_ == "retract") {
    Vec3 pos;
    Mat3 R;
    pose_for(*screw, taught_.screw_length + taught_.approach, &pos, &R);
    const Vec3 below(pos[0], pos[1], pos[2] - taught_.approach);
    const Vec6 qd =
        straight(q, below, pos, R, key_for("retract"), 45.0 * M_PI / 180.0, &left);
    if (left < 0.004) {
      index_ += 1;
      state_ = "wait";
      log_.push_back(StateLogEntry{round_to(frame.t, 2), "done", screw->name});
    }
    return qd;
  }

  if (state_ == "yield") {
    const std::string key = poses_.count(yield_key_) ? yield_key_ : std::string("park:park");
    const Vec6 qd = goto_pose(q, park_tip_, bit_down(), key, &left);
    if (left < 0.004 && clear_since_.has_value() && frame.t - *clear_since_ >= kClearForS) {
      const std::string resume = resume_state_.value_or("to_hole");
      yields_.push_back(YieldEntry{round_to(yield_since_, 2), round_to(frame.t, 2),
                                   screws_[static_cast<size_t>(index_)].name});
      // back into the cycle from the approach: whatever was half done is redone
      const bool at_the_hole = resume == "servo" || resume == "descend" || resume == "to_hole";
      go(at_the_hole ? "to_hole" : resume, frame.t);
      resume_state_.reset();
    }
    return qd;
  }

  return goto_pose(q, park_tip_, bit_down(), "park:park", &left);
}

// Waiting in place, over the box, is the worst place to wait: in gate 5 the worker
// walked into the held arm 54 times and it blocked the camera 44 % of the time. The
// park pose hides none of them (gate 1).
void B0::step_back(double t) {
  if (state_ == "wait" || state_ == "park" || state_ == "drive" ||
      index_ >= static_cast<int>(screws_.size())) {
    return;
  }
  if (state_ == "yield") {
    // held on the way to the stand-off as well: that is in his way too, go to park
    if (yield_key_ != "park:park") {
      forget(yield_key_);
      yield_key_ = "park:park";
    }
    return;
  }
  resume_state_ = state_;
  yield_since_ = t;
  clear_since_.reset();
  yield_key_ = "standoff:" + screws_[static_cast<size_t>(index_)].name;
  go("yield", t);
  forget(yield_key_);
}

// Only the voxels actually seen go into this test. It decides when to try again, not
// whether the move is safe: the motion layer still holds R1 against everything,
// unseen space included. A 30 cm keep-out round the hole was so strict that the arm
// waited at park until the cover went on and every rail screw was lost.
void B0::note_clear(double t, const Points &person, double safe) {
  if (index_ >= static_cast<int>(screws_.size())) {
    return;
  }
  const std::string key = "to_hole:" + screws_[static_cast<size_t>(index_)].name;
  const auto it = poses_.find(key);
  double gap = kInf;
  if (it != poses_.end() && person.rows() > 0) {
    gap = arm_gap(it->second, person);
  }
  if (gap < safe + kClearMarginM) {
    clear_since_.reset();
  } else if (!clear_since_.has_value()) {
    clear_since_ = t;
  }
}

double B0::arm_gap(const Vec6 &q, const Points &points) const {
  const mjModel *m = kin_->model();
  mjData *d = kin_->data();
  for (int j = 0; j < kNq; ++j) {
    d->qpos[j] = q[j];
  }
  mj_kinematics(m, d);
  double best = kInf;
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_group[g] != 3) {
      continue;
    }
    const Vec3 centre(d->geom_xpos[3 * g], d->geom_xpos[3 * g + 1], d->geom_xpos[3 * g + 2]);
    const auto [half, radius] = shape_line(m, g);
    const Vec3 axis =
        Vec3(d->geom_xmat[9 * g + 2], d->geom_xmat[9 * g + 5], d->geom_xmat[9 * g + 8]) * half;
    const Vec3 a = centre - axis;
    const Vec3 b = centre + axis;
    const Vec3 ab = b - a;
    const double denom = std::max(ab.dot(ab), 1e-12);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
      const Vec3 p = points.row(i).transpose();
      const double s = std::min(std::max((p - a).dot(ab) / denom, 0.0), 1.0);
      best = std::min(best, (p - (a + s * ab)).norm() - radius);
    }
  }
  return best;
}

Vec6 B0::move_to(const Vec6 &q, const Vec3 &pos, const Mat3 &R, double *error_out,
                 double tool_speed) const {
  return kin_->velocity_to(q, pos, &R, error_out, 2.0, 2.0, tool_speed, kMaxJointSpeed, 0.05);
}

// Interpolating the joints between the two ends instead bows the bit 1.8 mm sideways
// over a 60 mm descent, which would catch the screw on the hole edge.
Vec6 B0::straight(const Vec6 &q, const Vec3 &pos_from, const Vec3 &pos_to, const Mat3 &R,
                  const std::string &key, double speed, double *left_out, int steps) {
  auto found = plans_.find(key);
  if (found == plans_.end()) {
    Vec6 seed = q;
    std::vector<Vec6> plan;
    for (int i = 1; i <= steps; ++i) {
      const Vec3 point = pos_from + (static_cast<double>(i) / steps) * (pos_to - pos_from);
      // stay on the arm configuration we are already in: searching seeds again
      // can hop to another branch, and interpolating across a hop throws the
      // bit 37 mm sideways.
      double err = kInf;
      Vec6 next = kin_->ik(point, &R, seed, &err, 200);
      if (err > 1e-4) {
        next = kin_->ik_multi(point, &R, seed, &err);
      }
      if (!std::isfinite(err) || err > 1e-4) {
        unreachable_.push_back(key);
        break;
      }
      seed = next;
      plan.push_back(seed);
    }
    found = plans_.emplace(key, plan).first;
    leg_[key] = 0;
  }
  const std::vector<Vec6> &plan = found->second;
  if (plan.empty()) {
    *left_out = 0.0;
    return Vec6::Zero();
  }
  int leg = leg_.count(key) ? leg_[key] : 0;
  Vec6 left = plan[static_cast<size_t>(leg)] - q;
  // only the last point has to be hit exactly, the ones on the way just shape the path
  if (left.cwiseAbs().maxCoeff() < 0.03 && leg < static_cast<int>(plan.size()) - 1) {
    leg_[key] = leg + 1;
    left = plan[static_cast<size_t>(leg + 1)] - q;
  }
  const double done = (plan.back() - q).cwiseAbs().maxCoeff();
  // the speed is profiled on the distance to the last point, the direction on this leg
  const Vec6 direction = left / std::max(left.cwiseAbs().maxCoeff(), 1e-9);
  *left_out = done;
  return profiled(direction * done, key, speed, now_);
}

// Letting each joint ramp on its own instead made the small wrist joints finish while
// the shoulder was still starting its swing, which reached the tool out and down 40 cm
// below the path before it turned.
Vec6 B0::profiled(const Vec6 &left, const std::string &key, double speed, double t) {
  const double dt =
      !last_t_.has_value() ? 0.01 : std::min(std::max(t - *last_t_, 1e-3), 0.05);
  const double dist = left.cwiseAbs().maxCoeff();
  if (dist < 1e-9) {
    speed_[key] = 0.0;
    return Vec6::Zero();
  }
  const double stop = std::sqrt(2.0 * kJointAccel * dist);
  const double target = std::min({speed, stop, dist / dt});
  double now = speed_.count(key) ? speed_[key] : 0.0;
  now = target >= now ? std::min(target, now + kJointAccel * dt)
                      : std::max(target, now - 3 * kJointAccel * dt);
  speed_[key] = now;
  return (left / dist) * now;
}

// Driving the tool along a straight line instead walks joints into their limits and
// stops 17 mm short, which is where the first run of this baseline stuck.
Vec6 B0::goto_pose(const Vec6 &q, const Vec3 &pos, const Mat3 &R, const std::string &key,
                   double *left_out, double speed, const double *t) {
  auto found = goals_.find(key);
  if (found == goals_.end()) {
    const auto taught = poses_.find(key);
    Vec6 goal;
    if (taught != poses_.end()) {
      // exactly as taught. The teaching chain already decides which way each
      // joint turns; "fewest turns" would swing the shoulder through the front,
      // over the worker, where the taught route goes round the back.
      goal = taught->second;
    } else {
      double err = kInf;
      goal = kin_->ik_multi(pos, &R, q, &err);
      if (!std::isfinite(err)) {
        unreachable_.push_back(key);
        goals_[key] = q;
        *left_out = 0.0;
        return Vec6::Zero();
      }
    }
    found = goals_.emplace(key, goal).first;
  }
  const Vec6 left = found->second - q;
  const Vec6 qd = profiled(left, key, speed, t == nullptr ? now_ : *t);
  *left_out = left.cwiseAbs().maxCoeff();
  return qd;
}

void B0::forget(const std::string &key) {
  goals_.erase(key);
  plans_.erase(key);
  leg_.erase(key);
  speed_.erase(key);
}

void B0::go(const std::string &state, double t, double dwell) {
  forget(key_for(state));                   // solve the next move fresh
  state_ = state;
  until_ = t + dwell;
  log_.push_back(StateLogEntry{
      round_to(t, 2), state,
      index_ < static_cast<int>(screws_.size()) ? screws_[static_cast<size_t>(index_)].name : "-"});
}

std::string B0::key_for(const std::string &state) const {
  if (state == "park" || state == "yield") {
    return "park:park";
  }
  const std::string name = index_ < static_cast<int>(screws_.size())
                               ? screws_[static_cast<size_t>(index_)].name
                               : std::string("park");
  return state + ":" + name;
}

std::vector<std::string> B0::parts_in_jig(const std::string &signal, bool box_seen) const {
  if (index_ >= static_cast<int>(screws_.size()) || signal == "done") {
    return {};
  }
  if (signal == "cover_on" || signal == "cover_ready") {
    return {"wp_base", "wp_rail", "wp_cover"};
  }
  if (signal == "rail_ready") {
    return {"wp_base", "wp_rail"};
  }
  return box_seen ? std::vector<std::string>{"wp_base"} : std::vector<std::string>{};
}

Summary B0::summary() const {
  Summary out;
  for (const Screw &s : screws_) {
    if (s.driven) {
      out.screws_driven += 1;
    }
    out.turn_angles_deg[s.name] = s.turn_deg;
  }
  out.screws_total = static_cast<int>(screws_.size());
  out.hole_seen_frames = hole_seen_;
  out.hole_missed_frames = hole_missed_;
  out.unreachable = unreachable_;
  out.hole_rejected_corrections = hole_rejected_;
  out.yields = yields_;
  out.missed = missed_;
  out.hole_error_mm = hole_error_mm_;
  out.state_log = log_;
  return out;
}

}  // namespace sightline
