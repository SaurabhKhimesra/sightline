// Every 10 ms: would the arm's next moves put any part of it where it must not be?
//
// The arm is 63 points, each with its own margin: the joints, the tool tip, the wrist
// camera and a sample every 5 cm along each link. They are projected into the grid at
// the time the arm would be there, so a move that is clear now but would meet the
// person before the arm could stop is caught now.
//
// command() answers the hard question, whether the arm can keep this command one more
// cycle and still stop, and falls back through a slower command, braking and a way
// out. pose_ok() answers the planning question, whether a pose is reachable without
// getting any deeper than standing still would.
#ifndef SIGHTLINE_PLANNER_GUARD_HPP
#define SIGHTLINE_PLANNER_GUARD_HPP

#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "sightline_planner/kin.hpp"
#include "sightline_planner/motion.hpp"
#include "sightline_planner/types.hpp"
#include "sightline_planner/viewgrid.hpp"

namespace sightline {

constexpr double kSampleM = 0.05;                        // a point at least every 5 cm along each link (scene choice)
constexpr double kCycleS = 0.01;                         // the motion cycle
constexpr double kStopDecel = 1200.0 * M_PI / 180.0;     // the protective stop's deceleration per joint, 3 x the working ramp (scene choice)
constexpr double kEscapeSpeed = 90.0 * M_PI / 180.0;     // the joint speed limit the motion layer keeps
constexpr double kStepS = 0.04;                          // one check point per camera frame along a trajectory
// A point inside the margin may not get any deeper. Both sides of the comparison use the
// same grid at the same moment, so there is no noise to allow for; a 2 mm slack let a
// slow move creep down onto his hand 2 mm a cycle.
constexpr double kTolM = 1e-9;
constexpr double kEscapeGainM = 0.01;   // a way out goes first when it takes this much depth off the arm in 0.2 s (scene choice)
constexpr double kStaysM = 0.03;        // in planning, a point that moves less than this may stay where it is (scene choice)
constexpr double kHistoryS = 2.0;       // clear poses remembered for backing out the way the arm came
constexpr double kUpSpeed = 0.25;       // the tool straight up, toward the camera, m/s: the way out from anything below (scene choice)
constexpr double kBlindCellM = 0.05;    // the blind cells' size: a hand can be anywhere in one

// where an escape is scored, s ahead (scene choice)
inline constexpr double kEscapeSamples[2] = {0.10, 0.20};

// A pose the arm could run to when it has to get out of the way.
struct EscapeTarget {
  std::string name;
  Vec6 pose = Vec6::Zero();
  bool has_pose = false;
};

// How many moving points of the arm each check flags, and the deepest, for logs.
struct GuardExplain {
  int grid = 0;
  int blind = 0;
  int worktop = 0;
  double deepest_mm = 0.0;
  bool has_worst = false;
  Vec3 worst_point = Vec3::Zero();
  Vec3 his_cell = Vec3::Zero();
  bool cell_hidden = false;
  double cell_speed = 0.0;
  double cell_age = 0.0;
};

class TrajectoryGuard {
 public:
  TrajectoryGuard(ToolKinematics *kin, ViewGrid *grid, const std::vector<Plane> &planes = {});

  // Each joint, the tool tip, the wrist camera, and points along every link.
  Points points(const Vec6 &q) const;

  // The grid cells the arm covers at these joints.
  Grid shadow(const Vec6 &q) const;

  GuardExplain explain(const Vec6 &q, double at) const;
  int count(const Vec6 &q, double at) const;
  bool pose_clear(const Vec6 &q, double at) const;

  // For planning: every point is clear, or hardly moves from q_from and is no deeper
  // than it is there.
  bool pose_ok(const Vec6 &q, double at, const Vec6 &q_from) const;

  // True if every configuration is clear at its own time; else the first bad time.
  std::pair<bool, double> path_clear(const std::vector<Vec6> &configs,
                                     const std::vector<double> &times) const;

  // Keep qd for keep_s, then a synchronised protective stop. Configurations and times.
  static void stop_after(const Vec6 &q, const Vec6 &qd, double now, std::vector<Vec6> *configs,
                         std::vector<double> *times, double keep_s = kCycleS);

  // The straight joint line to goal, a sample every step rad of the fastest joint,
  // with the time the arm would reach each one at speed.
  static void line(const Vec6 &q, const Vec6 &goal, double speed, double now,
                   std::vector<Vec6> *configs, std::vector<double> *times,
                   double step = 0.06);

  // The closest command to target the joints can reach in one cycle at this rate.
  static Vec6 toward(const Vec6 &qd_prev, const Vec6 &target, double rate = kStopDecel);

  // The joint velocity to send this cycle, and why: go, slow, brake or escape.
  std::pair<Vec6, std::string> command(const Vec6 &q, const Vec6 &qd_prev, const Vec6 &qd_want,
                                       double now,
                                       const std::vector<EscapeTarget> &targets = {});

  void set_blind(const Points &blind) { blind_ = blind; }
  const Points &blind() const { return blind_; }
  const VecX &radii() const { return radii_; }
  const std::vector<char> &moves() const { return moves_; }
  int moving_points() const { return static_cast<int>(moves_idx_.size()); }
  // of the moving points, how many never leave the base's axis far enough to get out
  // of anyone's way
  int rooted_points() const {
    int n = 0;
    for (char flag : rooted_) {
      n += flag ? 1 : 0;
    }
    return n;
  }
  const std::string &last() const { return last_; }
  const std::map<std::string, int> &verdicts() const { return verdicts_; }
  const std::vector<double> &ms() const { return ms_; }
  const std::vector<Plane> &planes() const { return planes_; }
  const std::vector<char> &above_worktop() const { return above_worktop_; }
  int checks() const { return checks_; }

 private:
  // How far inside the off-limits space each point that moves is, in metres.
  VecX excess_of(const Points &pts, double at) const;
  std::vector<char> bad(const Points &pts, double at) const;

  // Score each way out by how much depth is left on the arm 0.1 and 0.2 s ahead,
  // and take the best. Braking in place is one way.
  struct EscapeChoice {
    double last_score = 0.0;
    double total_score = 0.0;
    Vec6 first = Vec6::Zero();
    std::string name;
  };
  EscapeChoice escape(const Vec6 &q, const Vec6 &qd_prev, double now,
                      const std::vector<EscapeTarget> &targets) const;

  // Head for goal (or stop, when there is none) at the protective rate; the total
  // depth inside the off-limits space at each scoring time.
  void rollout(const Vec6 &q, const Vec6 &qd_prev, const Vec6 *goal, bool up, double now,
               Vec6 *first, std::vector<double> *scores) const;

  // The latest pose the arm passed through that is still clear now.
  bool retrace(const Vec6 &q, double now, Vec6 *out) const;

  ToolKinematics *kin_;
  ViewGrid *grid_;
  std::vector<Plane> planes_;
  Points blind_ = Points(0, 3);          // blind cells that are live right now
  double blind_pad_;

  std::vector<int> joint_bodies_;
  std::vector<int> sample_geom_;         // the geom each link sample belongs to
  std::vector<double> sample_off_;       // and how far along its axis it sits
  int tip_ = -1;
  int cam_body_ = -1;
  Vec3 cam_local_ = Vec3::Zero();

  VecX radii_;
  std::vector<char> moves_;
  std::vector<int> moves_idx_;
  VecX radii_moves_;
  std::vector<char> above_worktop_;      // over every point
  std::vector<char> above_worktop_moves_;
  std::vector<char> rooted_;             // over the moving points only

  std::deque<std::pair<double, Vec6>> history_;   // (time, q) of poses that were clear when the arm was there
  mutable int checks_ = 0;
  std::map<std::string, int> verdicts_;
  std::string last_;
  std::vector<double> ms_;
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_GUARD_HPP
