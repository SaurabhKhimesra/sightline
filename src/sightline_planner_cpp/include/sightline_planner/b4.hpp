// B4: the rules, and a robot that looks for a way to keep working.
//
// It keeps several ways of driving every screw, one per usable turn angle around the
// bit, each solved once when the cell is taught. The choice is made when the arm
// leaves the feeder, with the latest picture:
//
// 1. a hole and turn angle whose approach pose, screwing pose and the joint path
//    there are all outside the grid's off-limits space;
// 2. of those, the shortest joint move, preferring a hole the wrist camera can see
//    past the robot's own links;
// 3. if none is clear, wait at the nearest clear stand-off and choose again with
//    every picture.
//
// The rest of the path is rechecked with every picture, and a hole whose way has
// closed is swapped before the arm gets there. At the hole, if the guard has to brake
// or move the arm, the screw is given up for now.
#ifndef SIGHTLINE_PLANNER_B4_HPP
#define SIGHTLINE_PLANNER_B4_HPP

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sightline_planner/b0.hpp"
#include "sightline_planner/guard.hpp"
#include "sightline_planner/viewgrid.hpp"

namespace sightline {

constexpr int kOptionStepDeg = 15;      // turn angles tried per screw when teaching (scene choice)
constexpr int kOptionsPerScrew = 24;    // how many ways of driving each screw are kept (scene choice)
constexpr double kSmoothLimitRad = 0.35;  // approach to hole joint move above this bows the bit: not kept
constexpr double kLookAheadS = 0.5;     // a block this close along the path is acted on now (scene choice)
// Planning asks the grid as it will be this far ahead, the guard's own stopping horizon.
// Asked about the grid as it is, a hole passed the choice and failed the guard's next
// check, the arm was held, gave the hole up and chose it again: twelve times in 12 s.
constexpr double kPlanAheadS = 0.15;
constexpr double kCoolDownS = 1.0;      // a hole given up is not chosen again for this long (scene choice)

bool at_the_hole(const std::string &state);

// One way of driving one screw, solved once when the cell is taught.
struct Option {
  double turn_deg = 0.0;
  Vec6 q_hole = Vec6::Zero();        // the approach pose above the hole
  Vec6 q_down = Vec6::Zero();        // the screwing pose, bit in the hole
  Vec6 q_standoff = Vec6::Zero();
  bool has_standoff = false;
  bool view_clear = false;           // the wrist camera sees the hole past the robot's own links
  double smooth = 0.0;               // joint move from the approach to the hole, rad
};

class B4 : public B0 {
 public:
  B4(ToolKinematics *kin, const Taught &taught, const Vec3 &park_tip, ViewGrid *grid,
     const std::vector<Plane> &planes = {});

  // Teach the cycle, the feeder pose and every way of driving every screw.
  void build(const Vec6 *home_q = nullptr, const Vec3 *eye = nullptr);

  // A new picture: rebuild the grid. q_capture is where the arm was when it was
  // taken, which is what hid part of the view.
  void see(const std::vector<int> &person_px, const std::vector<double> &person_z,
           double taken_at, const Vec6 &q_capture, const Points *blind = nullptr);

  // B2 and B3 decide when to try again from the voxels; B4 asks the grid instead.
  void note_clear(double t, const Points &person, double safe = 0.10) override;

  // Pick the hole to drive next and how. Returns false if nothing is clear.
  bool choose(const SensorFrame &frame, const std::string &why);

  // Held or moved by the guard at the hole: leave this screw for now and wait.
  void step_back(double t) override;

  // The guard's say on this cycle's command, and what the task does about it.
  Vec6 safe_command(const SensorFrame &frame, const Vec6 &qd_task);

  Summary summary() const override;

  TrajectoryGuard *guard() const { return guard_.get(); }
  ViewGrid *grid() const { return grid_; }
  double feeder_turn() const { return feeder_turn_; }
  const std::map<std::string, std::vector<Option>> &options() const { return options_; }
  const std::string &yield_key() const { return yield_key_; }

 protected:
  Vec6 task_inner(const SensorFrame &frame) override;

 private:
  // The feeder pose that keeps the whole arm behind the jig's back edge.
  void teach_feeder();

  // One way of driving each screw at every turn angle around the bit, solved once.
  void teach_options();

  // No part of the arm under the worktop, the shoulder aside.
  bool above_worktop(const Vec6 &q) const;

  // Is the line from the wrist camera to the hole clear of the robot's own links?
  bool wrist_sees(const Vec6 &q, const Vec3 &hole) const;

  // Travel time to the first pose on the joint line to goal that is off limits in the
  // grid as it is now, or infinity.
  double first_block(const Vec6 &q, const Vec6 &goal, double now) const;

  // The nearest stand-off, of any screw still to do, that is clear now; else park.
  std::string wait_pose(const Vec6 &q, double now);

  void swap_hole(const SensorFrame &frame, const std::string &why);
  void move_wait(const Vec6 &q, double now);
  void wait_somewhere(const SensorFrame &frame);

  std::optional<Vec3> eye_;                      // where the planner believes the eyes camera is
  std::map<std::string, std::vector<Option>> options_;
  std::vector<ChoiceEntry> choices_;             // for every choice made
  ViewGrid *grid_ = nullptr;                     // the camera's grid
  std::unique_ptr<TrajectoryGuard> guard_;
  std::optional<double> waiting_since_;
  std::optional<double> blocked_since_;
  std::vector<GaveUpEntry> gave_up_;             // for every screw left for now
  std::vector<SwapEntry> swaps_;                 // for every change of hole on the way
  std::optional<double> planned_on_;             // the picture the last path check used
  std::optional<Vec6> q_;
  double feeder_turn_ = 0.0;                     // the turn angle the feeder pose is taught at
  std::optional<double> feeder_clearance_;
  int closed_pictures_ = 0;                      // pictures in a row the way to the chosen hole looked closed
  std::map<std::pair<std::string, double>, double> dropped_;  // when a way was last given up
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_B4_HPP
