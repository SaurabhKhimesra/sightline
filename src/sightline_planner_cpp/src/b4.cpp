#include "sightline_planner/b4.hpp"

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

double percentile_of(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double pos = p / 100.0 * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return values[lo];
  }
  return values[lo] + (pos - lo) * (values[hi] - values[lo]);
}

}  // namespace

bool at_the_hole(const std::string &state) {
  return state == "servo" || state == "descend" || state == "drive" || state == "retract";
}

B4::B4(ToolKinematics *kin, const Taught &taught, const Vec3 &park_tip, ViewGrid *grid,
       const std::vector<Plane> &planes)
    : B0(kin, taught, park_tip), grid_(grid) {
  guard_ = std::make_unique<TrajectoryGuard>(kin, grid, planes);
}

void B4::build(const Vec6 *home_q, const Vec3 *eye) {
  B0::build(home_q);
  if (eye != nullptr) {
    eye_ = *eye;
  }
  teach_feeder();
  teach_options();
}

// The arm waits at the feeder for most of the cycle, so where it waits matters. The
// bit is round and the pick is straight down, so any turn angle will do: of every
// angle and arm configuration that reaches the feeder, keep the one whose arm stays
// furthest behind the jig's back edge. B0's taught pose hung the elbow over the right
// of the jig, where the parts go, and the worker pushed it off every cycle.
void B4::teach_feeder() {
  const auto park_it = poses_.find("park:park");
  if (park_it == poses_.end()) {
    return;
  }
  const Vec6 park = park_it->second;
  const Vec3 above = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.approach);
  const Vec3 pick = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.screw_length);
  const double back_edge = taught_.part_pose[1] + taught_.cover_size_y / 2;
  // the first screw's feeder pose, the one B0 taught first, as an extra seed
  std::vector<Vec6> old;
  for (const Screw &screw : screws_) {
    const auto it = poses_.find("to_feeder:" + screw.name);
    if (it != poses_.end()) {
      old.push_back(it->second);
      break;
    }
  }

  bool have = false;
  double best_behind = 0.0;
  double best_near_park = 0.0;
  double best_psi = 0.0;
  Vec6 best_q = Vec6::Zero();
  for (int psi = 0; psi < 360; psi += kOptionStepDeg) {
    const Mat3 R = rot_z(psi * M_PI / 180.0) * bit_down();
    for (const Vec6 &q : kin_->ik_all(above, &R, park, old)) {
      double err = kInf;
      const Vec6 q_dn = kin_->ik(pick, &R, q, &err, 200);
      if (err > 1e-4 || (q_dn - q).cwiseAbs().maxCoeff() > kSmoothLimitRad) {
        continue;
      }
      if (!(above_worktop(q) && above_worktop(q_dn))) {
        continue;
      }
      const Points pts = guard_->points(q);
      double behind = kInf;
      int moving = 0;
      for (Eigen::Index i = 0; i < pts.rows(); ++i) {
        if (!guard_->moves()[static_cast<size_t>(i)]) {
          continue;
        }
        behind = std::min(behind, pts(i, 1) - guard_->radii()[i]);
        moving += 1;
      }
      if (moving == 0) {
        continue;
      }
      behind -= back_edge;
      const double rounded = round_to(behind, 2);
      const double near_park = -(q - park).cwiseAbs().maxCoeff();
      if (!have || rounded > best_behind ||
          (rounded == best_behind && near_park > best_near_park)) {
        have = true;
        best_behind = rounded;
        best_near_park = near_park;
        best_psi = psi;
        best_q = q;
      }
    }
  }
  if (!have) {
    return;
  }
  feeder_turn_ = best_psi;
  for (auto &[key, pose] : poses_) {
    if (key.rfind("to_feeder:", 0) == 0) {
      pose = best_q;
    }
  }
  feeder_clearance_ = best_behind;
}

// Which side of the hole the wrist ends up on depends on the turn angle alone: at the
// first rail screw, 0 to 75 and 270 to 330 degrees keep both wrist joints on the
// robot's side and the rest put them 5 to 19 cm beyond the hole toward the worker.
// Each angle is solved from several starting poses; the configuration nearest the
// feeder pose is kept if it goes down into the hole smoothly and no part of the arm
// ends up under the worktop.
void B4::teach_options() {
  const auto ref_it = poses_.find("park:park");
  for (const Screw &screw : screws_) {
    const auto seed_it = poses_.find("to_hole:" + screw.name);
    if (seed_it == poses_.end()) {
      continue;
    }
    const Vec6 q_seed = seed_it->second;
    const auto start_it = poses_.find("to_feeder:" + screw.name);
    const Vec6 start = start_it == poses_.end() ? q_seed : start_it->second;
    const Vec3 above = screw.hole + Vec3(0.0, 0.0, taught_.screw_length + taught_.approach);
    const Vec3 down = screw.hole + Vec3(0.0, 0.0, taught_.screw_length);
    const Vec3 back = screw.hole + Vec3(0.0, kStandoffBackM, kStandoffUpM);
    std::vector<Option> found;
    for (int psi = 0; psi < 360; psi += kOptionStepDeg) {
      const Mat3 R = rot_z(psi * M_PI / 180.0) * bit_down();
      std::vector<Vec6> seeds{q_seed};
      if (ref_it != poses_.end()) {
        seeds.push_back(ref_it->second);
      }
      for (const Vec6 &q_up : kin_->ik_all(above, &R, start, seeds)) {
        double err = kInf;
        const Vec6 q_dn = kin_->ik(down, &R, q_up, &err, 200);
        if (err > 1e-4) {
          continue;
        }
        const double smooth = (q_dn - q_up).cwiseAbs().maxCoeff();
        if (smooth > kSmoothLimitRad || !(above_worktop(q_up) && above_worktop(q_dn))) {
          continue;
        }
        Vec6 q_wait = kin_->ik(back, &R, q_up, &err, 300);
        if (err > 1e-4) {
          q_wait = kin_->ik_multi(back, &R, q_up, &err);
        }
        Option option;
        option.turn_deg = psi;
        option.q_hole = q_up;
        option.q_down = q_dn;
        option.has_standoff = std::isfinite(err) && above_worktop(q_wait);
        option.q_standoff = q_wait;
        option.view_clear = wrist_sees(q_up, screw.hole);
        option.smooth = smooth;
        found.push_back(option);
        break;
      }
    }
    std::stable_sort(found.begin(), found.end(), [&start](const Option &a, const Option &b) {
      return (a.q_hole - start).cwiseAbs().maxCoeff() < (b.q_hole - start).cwiseAbs().maxCoeff();
    });
    if (static_cast<int>(found.size()) > kOptionsPerScrew) {
      found.resize(kOptionsPerScrew);
    }
    options_[screw.name] = found;
  }
}

bool B4::above_worktop(const Vec6 &q) const {
  if (guard_->planes().empty()) {
    return true;
  }
  const Points pts = guard_->points(q);
  for (const Plane &plane : guard_->planes()) {
    for (Eigen::Index i = 0; i < pts.rows(); ++i) {
      if (!guard_->above_worktop()[static_cast<size_t>(i)]) {
        continue;
      }
      if ((pts.row(i).transpose() - plane.point).dot(plane.normal) < guard_->radii()[i]) {
        return false;
      }
    }
  }
  return true;
}

// Gate 4 found the upper arm and the forearm in that line at some turn angles, never
// the tool itself, which sits beside the camera on purpose.
bool B4::wrist_sees(const Vec6 &q, const Vec3 &hole) const {
  const auto [cam, cam_R] = kin_->camera_pose(q);
  const mjModel *m = kin_->model();
  mjData *d = kin_->data();
  for (int j = 0; j < kNq; ++j) {
    d->qpos[j] = q[j];
  }
  mj_kinematics(m, d);
  const int wrist = mj_name2id(m, mjOBJ_BODY, "ur5e/wrist_3_link");
  const Vec3 a = cam;
  const Vec3 b = hole;
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_group[g] != 3 || m->geom_bodyid[g] == wrist) {
      continue;
    }
    const Vec3 centre(d->geom_xpos[3 * g], d->geom_xpos[3 * g + 1], d->geom_xpos[3 * g + 2]);
    const auto [half, radius] = shape_line(m, g);
    const Vec3 axis =
        Vec3(d->geom_xmat[9 * g + 2], d->geom_xmat[9 * g + 5], d->geom_xmat[9 * g + 8]) * half;
    const Vec3 c0 = centre - axis;
    const Vec3 c1 = centre + axis;
    const Vec3 ab = c1 - c0;
    const double denom = ab.dot(ab);
    for (int k = 0; k < 12; ++k) {
      const Vec3 p = a + (k / 11.0) * (b - a);
      const double t =
          denom < 1e-12 ? 0.0 : std::min(std::max((p - c0).dot(ab) / denom, 0.0), 1.0);
      if ((p - (c0 + t * ab)).norm() < radius) {
        return false;
      }
    }
  }
  return true;
}

void B4::see(const std::vector<int> &person_px, const std::vector<double> &person_z,
             double taken_at, const Vec6 &q_capture, const Points *blind) {
  const Grid shadow = guard_->shadow(q_capture);
  grid_->update(person_px, person_z, taken_at, &shadow);
  if (blind != nullptr) {
    guard_->set_blind(*blind);
  }
}

void B4::note_clear(double, const Points &, double) {}

double B4::first_block(const Vec6 &q, const Vec6 &goal, double now) const {
  std::vector<Vec6> configs;
  std::vector<double> arrive;
  TrajectoryGuard::line(q, goal, kMaxJointSpeed, now, &configs, &arrive);
  for (size_t i = 0; i < configs.size(); ++i) {
    if (!guard_->pose_ok(configs[i], now + kPlanAheadS, q)) {
      return arrive[i] - now;
    }
  }
  return kInf;
}

bool B4::choose(const SensorFrame &frame, const std::string &why) {
  const Vec6 q = frame.q;
  const double now = frame.t;
  const int stage = stage_index(frame.jig_signal);
  struct Candidate {
    double cost;
    int k;
    const Option *option;
  };
  std::vector<Candidate> candidates;
  for (int k = index_; k < static_cast<int>(screws_.size()); ++k) {
    const Screw &screw = screws_[static_cast<size_t>(k)];
    if (screw.driven || stage_index(screw.needs) != stage) {
      continue;
    }
    const auto it = options_.find(screw.name);
    if (it == options_.end()) {
      continue;
    }
    for (const Option &opt : it->second) {
      const double cost =
          (opt.q_hole - q).cwiseAbs().maxCoeff() + (opt.view_clear ? 0.0 : 1.0);
      candidates.push_back(Candidate{cost, k, &opt});
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate &a, const Candidate &b) { return a.cost < b.cost; });
  for (const Candidate &candidate : candidates) {
    const Option &opt = *candidate.option;
    const int k = candidate.k;
    const auto dropped = dropped_.find({screws_[static_cast<size_t>(k)].name, opt.turn_deg});
    if (dropped != dropped_.end() && now - dropped->second < kCoolDownS) {
      continue;
    }
    const double ahead = now + kPlanAheadS;
    if (!(guard_->pose_ok(opt.q_hole, ahead, q) && guard_->pose_ok(opt.q_down, ahead, q))) {
      continue;
    }
    if (std::isfinite(first_block(q, opt.q_hole, now))) {
      continue;
    }
    // bring the chosen screw to the front of the queue and drive it this way
    const Option kept = opt;
    std::swap(screws_[static_cast<size_t>(index_)], screws_[static_cast<size_t>(k)]);
    Screw &screw = screws_[static_cast<size_t>(index_)];
    screw.turn_deg = kept.turn_deg;
    poses_["to_hole:" + screw.name] = kept.q_hole;
    if (kept.has_standoff) {
      poses_["standoff:" + screw.name] = kept.q_standoff;
    }
    choices_.push_back(ChoiceEntry{round_to(now, 2), screw.name, kept.turn_deg, why});
    return true;
  }
  return false;
}

std::string B4::wait_pose(const Vec6 &q, double now) {
  struct Found {
    double gap;
    std::string name;
    Vec6 pose;
  };
  std::vector<Found> found;
  for (size_t i = static_cast<size_t>(index_); i < screws_.size(); ++i) {
    const Screw &screw = screws_[i];
    if (screw.driven) {
      continue;
    }
    const auto it = options_.find(screw.name);
    if (it == options_.end()) {
      continue;
    }
    for (const Option &opt : it->second) {
      if (opt.has_standoff) {
        found.push_back(
            Found{(opt.q_standoff - q).cwiseAbs().maxCoeff(), screw.name, opt.q_standoff});
      }
    }
  }
  std::stable_sort(found.begin(), found.end(),
                   [](const Found &a, const Found &b) { return a.gap < b.gap; });
  for (const Found &entry : found) {
    if (guard_->pose_ok(entry.pose, now, q)) {
      const std::string key = "standoff:" + entry.name + ":wait";
      poses_[key] = entry.pose;
      return key;
    }
  }
  return "park:park";
}

Vec6 B4::task_inner(const SensorFrame &frame) {
  const Vec6 q = frame.q;
  q_ = q;
  const double now = frame.t;
  drop_passed(frame);
  // pushed off the feeder by the guard: go back with a joint move, not the 10 cm/s
  // final approach, which took 7 s to come back from an escape
  if (state_ == "pick" && index_ < static_cast<int>(screws_.size())) {
    const Vec3 low = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.screw_length);
    const Vec3 high = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.approach);
    const Vec3 pos = kin_->fk(q).first;
    // off the short vertical line the pick runs along
    const Vec3 span = high - low;
    const double along =
        std::min(std::max((pos - low).dot(span) / std::max(span.dot(span), 1e-9), 0.0), 1.0);
    if ((pos - (low + along * span)).norm() > 0.03) {
      go("to_feeder", now);
    }
  }
  const bool new_picture =
      grid_->has_picture() && (!planned_on_.has_value() || grid_->taken_at() != *planned_on_);
  if (new_picture) {
    planned_on_ = grid_->taken_at();
  }
  // at the feeder: take a screw, and wait with it until its part is in. Then choose
  // the hole, not before
  if (state_ == "pick" && index_ < static_cast<int>(screws_.size())) {
    const Vec3 target = taught_.feeder_pick + Vec3(0.0, 0.0, taught_.screw_length);
    const Mat3 R = rot_z(feeder_turn_ * M_PI / 180.0) * bit_down();
    double err = kInf;
    const Vec6 qd = move_to(q, target, R, &err, 0.10);
    if (now >= until_ && err < 0.004 &&
        frame.jig_signal == screws_[static_cast<size_t>(index_)].needs) {
      if (choose(frame, "left the feeder")) {
        go("to_hole", now);
      } else {
        wait_somewhere(frame);
      }
      return Vec6::Zero();
    }
    return qd;
  }
  if (new_picture && index_ < static_cast<int>(screws_.size())) {
    // waiting: choose again with every picture, go as soon as any hole is clear
    if (state_ == "yield") {
      if (choose(frame, "clear again")) {
        yields_.push_back(YieldEntry{round_to(yield_since_, 2), round_to(now, 2),
                                     screws_[static_cast<size_t>(index_)].name});
        waiting_since_.reset();
        go("to_hole", now);
        return Vec6::Zero();
      }
      if (yield_key_ != "park:park" && !guard_->pose_ok(poses_[yield_key_], now, q)) {
        move_wait(q, now);
      }
    } else if (state_ == "to_hole") {
      // Only when it stays closed for two pictures: a path that grazes the
      // margin flips between clear and closed with the noise, and B4 chose
      // and dropped the same hole twelve times in 12 s. The guard's own check
      // every 10 ms still holds the arm the moment he really comes close.
      const auto goal = poses_.find(key_for("to_hole"));
      if (goal != poses_.end() && first_block(q, goal->second, now) < kLookAheadS) {
        closed_pictures_ += 1;
        if (closed_pictures_ >= 2) {
          closed_pictures_ = 0;
          swap_hole(frame, "the way closed");
        }
      } else {
        closed_pictures_ = 0;
      }
    }
  }
  // waiting for the next part: fetch its screw now and wait at the feeder with it.
  // Waiting at park instead meant a 180 degree swing round the back of the base
  // once the part was in, while he was still placing it 24 cm in front of the
  // base, and the swing was stopped for most of the 14 s the rail screws allow.
  if (state_ == "wait" && index_ < static_cast<int>(screws_.size())) {
    go("to_feeder", now);
    return Vec6::Zero();
  }
  return base_task_inner(frame);
}

void B4::swap_hole(const SensorFrame &frame, const std::string &why) {
  const std::string was = screws_[static_cast<size_t>(index_)].name;
  dropped_[{was, screws_[static_cast<size_t>(index_)].turn_deg}] = frame.t;
  if (choose(frame, why)) {
    swaps_.push_back(SwapEntry{round_to(frame.t, 2), was, why});
    go("to_hole", frame.t);
  } else {
    wait_somewhere(frame);
  }
}

void B4::move_wait(const Vec6 &q, double now) {
  const std::string key = wait_pose(q, now);
  if (key != yield_key_) {
    yield_key_ = key;
    goals_.erase(key);
    speed_.erase(key);
  }
}

void B4::wait_somewhere(const SensorFrame &frame) {
  resume_state_ = "to_hole";
  yield_since_ = frame.t;
  waiting_since_ = frame.t;
  yield_key_ = wait_pose(frame.q, frame.t);
  go("yield", frame.t);
  goals_.erase(yield_key_);
  speed_.erase(yield_key_);
}

void B4::step_back(double t) {
  if (index_ >= static_cast<int>(screws_.size())) {
    return;
  }
  const Screw &screw = screws_[static_cast<size_t>(index_)];
  if (state_ == "retract") {
    // the screw is in; only the way out was cut short, and the guard is seeing to that
    index_ += 1;
    state_ = "wait";
    log_.push_back(StateLogEntry{round_to(t, 2), "done", screw.name});
    return;
  }
  if (!(state_ == "to_hole" || at_the_hole(state_))) {
    return;
  }
  gave_up_.push_back(GaveUpEntry{round_to(t, 2), screw.name, state_});
  dropped_[{screw.name, screw.turn_deg}] = t;
  const Vec6 q = q_.has_value() ? *q_ : poses_.at("park:park");
  resume_state_ = "to_hole";
  yield_since_ = t;
  waiting_since_ = t;
  yield_key_ = wait_pose(q, t);
  go("yield", t);
  goals_.erase(yield_key_);
  speed_.erase(yield_key_);
}

Vec6 B4::safe_command(const SensorFrame &frame, const Vec6 &qd_task) {
  // at the hole the way out is the screw's stand-off; anywhere else, back and up
  // are nearer than park, which is on the far side of the base
  std::vector<EscapeTarget> targets;
  auto add = [&](const std::string &name, const std::string &key) {
    const auto it = poses_.find(key);
    EscapeTarget target;
    target.name = name;
    target.has_pose = it != poses_.end();
    if (target.has_pose) {
      target.pose = it->second;
    }
    targets.push_back(target);
  };
  if ((state_ == "to_hole" || at_the_hole(state_)) &&
      index_ < static_cast<int>(screws_.size())) {
    add("standoff", "standoff:" + screws_[static_cast<size_t>(index_)].name);
  } else if (state_ == "yield") {
    add("standoff", yield_key_);
  }
  add("park", "park:park");

  const auto [qd, verdict] = guard_->command(frame.q, frame.qd, qd_task, frame.t, targets);
  const bool blocked = verdict == "brake" || verdict.rfind("escape", 0) == 0;
  if (!blocked) {
    blocked_since_.reset();
    return qd;
  }
  if (!blocked_since_.has_value()) {
    blocked_since_ = frame.t;
  }
  if (at_the_hole(state_)) {
    step_back(frame.t);
  } else if (state_ == "to_hole" && frame.t - *blocked_since_ >= kYieldAfterS) {
    swap_hole(frame, "held on the way");
    blocked_since_.reset();
  }
  return qd;
}

Summary B4::summary() const {
  Summary out = B0::summary();
  out.choices = choices_;
  for (const auto &[name, list] : options_) {
    out.options_per_screw[name] = static_cast<int>(list.size());
    int clear = 0;
    for (const Option &opt : list) {
      clear += opt.view_clear ? 1 : 0;
    }
    out.options_view_clear[name] = clear;
  }
  out.gave_up = gave_up_;
  out.swaps = swaps_;
  if (guard_ != nullptr) {
    std::vector<double> ms = guard_->ms();
    if (ms.empty()) {
      ms.push_back(0.0);
    }
    out.has_guard = true;
    out.guard_verdicts = guard_->verdicts();
    out.guard_ms_median = round_to(percentile_of(ms, 50.0), 3);
    out.guard_ms_p99 = round_to(percentile_of(ms, 99.0), 3);
    out.guard_ms_max = round_to(*std::max_element(ms.begin(), ms.end()), 3);
    out.points_on_the_arm = static_cast<int>(guard_->radii().size());
  }
  out.feeder_turn_deg = feeder_turn_;
  out.has_feeder_clearance = feeder_clearance_.has_value();
  out.feeder_arm_behind_jig_m = feeder_clearance_.value_or(0.0);
  return out;
}

}  // namespace sightline
