// B0: visual servoing, no avoidance. The baseline that shows the problem.
//
// It drives every screw as a taught cell would: go to the feeder, take a screw, go to
// the hole, line up on the hole with the wrist camera, descend, drive, retract. It
// never looks for the person, because nothing in a plain cell does. Gate 3 measures
// what that costs.
//
// Speeds, dwell times and the stand-off are scene choices and are stated in
// docs/notes.md. The robot's own limits come from the UR5e datasheet.
#ifndef SIGHTLINE_PLANNER_B0_HPP
#define SIGHTLINE_PLANNER_B0_HPP

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "sightline_planner/frame.hpp"
#include "sightline_planner/kin.hpp"
#include "sightline_planner/motion.hpp"
#include "sightline_planner/types.hpp"

namespace sightline {

constexpr int kTurnStepDeg = 15;
constexpr double kMaxJointSpeed = 90.0 * M_PI / 180.0;   // scene choice, below the datasheet's 180 deg/s
constexpr double kMaxToolSpeed = 0.25;                   // m/s, scene choice
constexpr double kDriveS = 1.2;                          // how long a screw takes to go in, scene choice
constexpr double kPickS = 0.4;                           // dwell at the feeder, scene choice
constexpr double kReachedM = 0.0015;                     // close enough to call a move done, scene choice
// A correction bigger than this is not believed. The wrist detector is a centroid of
// dark pixels and lands 5 to 12 mm off the hole, which is worse than the taught
// position, so in gate 3 every correction is refused. Gate 4 builds the detector
// properly and this is where its output will be trusted.
constexpr double kMaxCorrectionM = 0.004;
constexpr double kJointAccel = 400.0 * M_PI / 180.0;     // the working ramp, the same number the motion layer enforces
constexpr double kYieldAfterS = 0.3;    // held this long by the rules near the work: step back (scene choice)
constexpr double kClearMarginM = 0.02;  // clear: the arm at the hole would stay one voxel beyond R1's distance (scene choice)
constexpr double kClearForS = 0.5;      // and has stayed clear this long (scene choice)
constexpr double kStandoffBackM = 0.25; // a screw's waiting pose: this far back toward the robot (scene choice)
constexpr double kStandoffUpM = 0.30;   // and this far above the hole (scene choice)

Mat3 rot_z(double a);

// The bit pointing straight down at the worktop.
const Mat3 &bit_down();

struct Screw {
  std::string name;
  Vec3 hole = Vec3::Zero();     // world position of the hole mouth
  std::string needs;            // the jig signal that has to be showing
  double turn_deg = 0.0;
  bool driven = false;
};

// One line of a log the runner writes out.
struct StateLogEntry {
  double t = 0.0;
  std::string state;
  std::string screw;
};

struct YieldEntry {
  double from = 0.0;
  double to = 0.0;
  std::string screw;
};

struct MissedEntry {
  double t = 0.0;
  std::string screw;
  std::string why;
};

struct ChoiceEntry {
  double t = 0.0;
  std::string screw;
  double turn_deg = 0.0;
  std::string why;
};

struct GaveUpEntry {
  double t = 0.0;
  std::string screw;
  std::string state;
};

struct SwapEntry {
  double t = 0.0;
  std::string screw;
  std::string why;
};

// What the runner writes out at the end of an episode. The fields after the guard
// are B4's and stay empty for the baseline.
struct Summary {
  int screws_driven = 0;
  int screws_total = 0;
  std::map<std::string, double> turn_angles_deg;
  int hole_seen_frames = 0;
  int hole_missed_frames = 0;
  std::vector<std::string> unreachable;
  int hole_rejected_corrections = 0;
  std::vector<YieldEntry> yields;
  std::vector<MissedEntry> missed;
  std::vector<double> hole_error_mm;
  std::vector<StateLogEntry> state_log;

  std::vector<ChoiceEntry> choices;
  std::map<std::string, int> options_per_screw;
  std::map<std::string, int> options_view_clear;
  std::vector<GaveUpEntry> gave_up;
  std::vector<SwapEntry> swaps;
  bool has_guard = false;
  std::map<std::string, int> guard_verdicts;
  double guard_ms_median = 0.0;
  double guard_ms_p99 = 0.0;
  double guard_ms_max = 0.0;
  int points_on_the_arm = 0;
  double feeder_turn_deg = 0.0;
  bool has_feeder_clearance = false;
  double feeder_arm_behind_jig_m = 0.0;
};

// State machine and motion for the no-avoidance baseline.
class B0 {
 public:
  // what the jig's switches can say, in the order they happen: rail pressed in, cover
  // placed (the rail screws are under it now), clamps closed, clamps opened again
  static const std::vector<std::string> &stages();
  static int stage_index(const std::string &signal);

  B0(ToolKinematics *kin, const Taught &taught, const Vec3 &park_tip);
  virtual ~B0() = default;

  // Make the screws, choose each one's turn angle, and teach the whole cycle.
  void build(const Vec6 *home_q = nullptr);

  // One motion cycle. With a motion layer attached the command it wants goes through
  // the layer, which keeps the rules; without one this is the baseline.
  Vec6 update(const SensorFrame &frame, const Obstacles *obstacles = nullptr);

  Vec6 task(const SensorFrame &frame);

  // The rules have held the arm near the work: retract to park and wait there.
  virtual void step_back(double t);

  // Called every frame with what the camera sees: could the arm be at the current
  // hole right now and keep R1's distance with a margin?
  virtual void note_clear(double t, const Points &person, double safe = 0.10);

  // Which parts the planner's own task state says are in the jig. It gets no
  // signal for the base alone, so the base counts once the box fit has seen it.
  std::vector<std::string> parts_in_jig(const std::string &signal, bool box_seen) const;

  virtual Summary summary() const;

  // Where the wrist camera says the hole is, in the cell frame: predict it in the
  // image from the taught pose, take the dark blob around that prediction, and put its
  // centroid back on the part's top face.
  std::optional<Vec3> find_hole(const SensorFrame &frame, const Screw &screw);

  const std::string &state() const { return state_; }
  int index() const { return index_; }
  const std::vector<Screw> &screws() const { return screws_; }
  std::vector<Screw> &screws() { return screws_; }
  const std::map<std::string, Vec6> &poses() const { return poses_; }
  double yield_since() const { return yield_since_; }
  void set_motion(MotionLayer *motion) { motion_ = motion; }
  MotionLayer *motion() const { return motion_; }
  const std::vector<Report> &reports() const { return reports_; }

 protected:
  void teach(const Vec6 &home);

  // The turn angle around the bit that gives the smoothest insertion.
  double pick_turn(const Vec3 &hole) const;

  void pose_for(const Screw &screw, double height, Vec3 *pos, Mat3 *R) const;

  // A screw whose stage has passed cannot be driven any more: once the cover is on, a
  // rail screw under it is out of reach. Record it as missed and move on.
  void drop_passed(const SensorFrame &frame);

  virtual Vec6 task_inner(const SensorFrame &frame);
  Vec6 base_task_inner(const SensorFrame &frame);

  // Closest distance from the arm at these joints to any of the points.
  double arm_gap(const Vec6 &q, const Points &points) const;

  // Fine Cartesian motion, for lining up and going in.
  Vec6 move_to(const Vec6 &q, const Vec3 &pos, const Mat3 &R, double *error_out,
               double tool_speed = kMaxToolSpeed) const;

  // A straight line for the tool tip: solve waypoints along it and step through them.
  Vec6 straight(const Vec6 &q, const Vec3 &pos_from, const Vec3 &pos_to, const Mat3 &R,
                const std::string &key, double speed, double *left_out, int steps = 5);

  // One speed for all joints, along the straight joint line to the goal: the fastest
  // joint ramps, cruises and slows to stop exactly at the goal, and every other joint
  // keeps its share of that speed.
  Vec6 profiled(const Vec6 &left, const std::string &key, double speed, double t);

  // A joint move to a solved pose, the way a taught cell gets from A to B.
  Vec6 goto_pose(const Vec6 &q, const Vec3 &pos, const Mat3 &R, const std::string &key,
                 double *left_out, double speed = kMaxJointSpeed, const double *t = nullptr);

  void go(const std::string &state, double t, double dwell = 0.0);
  std::string key_for(const std::string &state) const;
  void forget(const std::string &key);

  ToolKinematics *kin_;
  Taught taught_;
  Vec3 park_tip_;
  std::vector<Screw> screws_;
  std::string state_ = "wait";
  int index_ = 0;
  double until_ = 0.0;
  std::vector<StateLogEntry> log_;
  int hole_seen_ = 0;
  int hole_missed_ = 0;
  std::vector<std::string> unreachable_;
  std::vector<double> hole_error_mm_;
  int hole_rejected_ = 0;
  std::optional<Vec3> corrected_;
  std::map<std::string, Vec6> goals_;
  std::map<std::string, std::vector<Vec6>> plans_;
  std::map<std::string, int> leg_;
  MotionLayer *motion_ = nullptr;          // a motion layer, or none for the baseline
  std::vector<Report> reports_;
  std::map<std::string, double> speed_;
  std::optional<double> last_t_;
  double now_ = 0.0;
  std::map<std::string, Vec6> poses_;      // the taught joint poses, solved once
  std::optional<std::string> resume_state_;
  std::string yield_key_ = "park:park";
  double yield_since_ = 0.0;
  std::optional<double> clear_since_;
  std::vector<YieldEntry> yields_;         // for every step back
  std::vector<MissedEntry> missed_;        // for every screw given up
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_B0_HPP
