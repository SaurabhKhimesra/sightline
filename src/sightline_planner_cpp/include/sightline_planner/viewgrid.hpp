// The camera's view of the cell as a grid of directions, and where the robot may not be.
//
// Every direction in which the camera sees the person is off limits to the robot,
// from the camera out to just behind them, plus a margin: the safety distance, the
// size of the part of the arm being checked, and how far they could have moved since
// the picture. The grid is rebuilt from every picture at 25 Hz and the arm is checked
// against it every 10 ms.
//
// Cells the arm itself hides are kept for half a second at the depth they were last
// seen, and each cell carries its own measured speed. docs/notes.md has the numbers
// behind both and what the earlier versions got wrong.
#ifndef SIGHTLINE_PLANNER_VIEWGRID_HPP
#define SIGHTLINE_PLANNER_VIEWGRID_HPP

#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "sightline_planner/calibration.hpp"
#include "sightline_planner/types.hpp"

namespace sightline {

constexpr int kCellPx = 4;       // a grid cell is 4 by 4 pixels, about 2.3 cm at 3.2 m (scene choice)
constexpr int kMinPixels = 2;    // a cell needs two pixels of him: one stray pixel is noise (scene choice)
// A cell with no occupied neighbour is noise too. On the eight work poses, three noisy
// pictures each, this took the cells nowhere near him from 18.9 to 1.75 a picture and
// lost none of his (40.46 against 40.42 cells missed of 474, the ones the 40 mm
// background gate already hides). Three pixels a cell instead lost 15 more of his.
constexpr int kSolidCells = 6;   // speed is read only where a 5 by 5 cell window holds this much of him (scene choice)
constexpr double kBehindM = 0.15;   // behind a seen surface a hand can still hide: a limb's thickness (scene choice)
constexpr double kSafeM = 0.10;     // R1's safety distance, the same as the motion layer (scene choice)
constexpr int kSpeedBaseline = 3;   // speed is measured against the picture this many frames back (scene choice)
constexpr int kSpeedRank = 3;       // for logs: his third fastest cell
constexpr double kSpeedHoldS = 0.2; // each cell keeps the fastest speed of the last 0.2 s (scene choice)
constexpr double kPartM = 0.15;     // a moving part of him, a hand and wrist, shares its leading edge's speed (scene choice)
// ISO 13855's person speed, 2000 mm/s, as reproduced by Marvel and Norcross 2017,
// Sec. 3, pp. 146 and 148. The constant setting of docs/design.md section 8.6 (a).
constexpr double kIsoSpeed = 2.0;
constexpr double kSpeedCap = kIsoSpeed;  // a reading faster than the standard's own worst case is noise, not him
constexpr double kMemoryS = 0.5;         // a hidden cell is kept this long at most: the guard has moved the arm by then (scene choice)

// How far inside the off-limits space each asked point is, and which cell of him said so.
struct Excess {
  VecX depth;                // metres inside; negative is clear by that much
  std::vector<int> cells;    // index into the grid's cells, or -1
};

// Off-limits directions and depths, rebuilt from each picture of the person.
class ViewGrid {
 public:
  explicit ViewGrid(const Calibration &calib, int cell_px = kCellPx, double safe = kSafeM,
                    double behind = kBehindM, const std::string &speed_mode = "measured",
                    double accel = 0.0,
                    double timed_above = -std::numeric_limits<double>::infinity());

  const Calibration &calib() const { return calib_; }
  int cell() const { return cell_; }
  int gw() const { return gw_; }
  int gh() const { return gh_; }
  double safe() const { return safe_; }
  int size() const { return static_cast<int>(rows_.size()); }
  const std::vector<int> &rows() const { return rows_; }
  const std::vector<int> &cols() const { return cols_; }
  const std::vector<double> &near() const { return near_; }
  const std::vector<double> &far() const { return far_; }
  const std::vector<double> &seen_at() const { return seen_at_; }
  const std::vector<double> &cell_speed() const { return cell_speed_; }
  const std::vector<char> &hidden() const { return hidden_; }
  int remembered() const { return remembered_; }
  double speed() const { return speed_; }
  double measured() const { return measured_; }
  bool has_picture() const { return has_picture_; }
  double taken_at() const { return taken_at_; }

  // How far he could have moved in this long, at this speed.
  double reach(double age, double speed) const;
  double reach(double age) const { return reach(age, speed_); }

  // The empty cell's depth image: nothing of him can be behind it. No return reads
  // as unlimited.
  void set_background(const Grid &depth);

  // person_px are pixel indices of him in the picture, person_z their depths.
  // shadow is, per cell, how far out the arm's own surface was when the picture was
  // taken, infinite where the arm was not in the way.
  void update(const std::vector<int> &person_px, const std::vector<double> &person_z,
              double taken_at, const Grid *shadow = nullptr);

  // Where the cells he is seen in are, in the world, at their near depth.
  Points seen_points() const;

  // Where cell k of him is, in the world, at its near depth. For logs.
  Vec3 cell_point(int k) const;

  // The speed the grid gives the part of him nearest this point, for logs.
  double speed_near(const Vec3 &point, double within = 0.10) const;

  // Pixel column, pixel row and depth along the axis of each point; in front of the camera.
  void project(const Points &points, VecX *u, VecX *v, VecX *depth,
               std::vector<char> *ahead) const;

  // How far inside the off-limits space each point is, in metres; negative is clear
  // by that much, and minus infinity is nowhere near them. radii is how far the
  // robot's surface reaches beyond each point, now is one time per point.
  Excess excess(const Points &points, const VecX &radii, const VecX &now,
                bool want_cells = false) const;
  Excess excess(const Points &points, const VecX &radii, double now,
                bool want_cells = false) const;

  // True for every point the robot may not occupy at time now.
  std::vector<char> forbidden(const Points &points, const VecX &radii, double now) const;

  // Per cell, how far out from the camera the arm's own surface is: whatever is
  // behind it is hidden. Infinite where the arm is not in the way.
  Grid shadow_of(const Points &points, const VecX &radii) const;

  // Cell centres at these depths, in the camera's frame: distances are what matter.
  Points points_at(const std::vector<int> &rows, const std::vector<int> &cols,
                   const std::vector<double> &z) const;

 private:
  Grid measure_speed(const GridB &seen, const Grid &near, double t);

  Calibration calib_;
  int cell_;
  int gw_;
  int gh_;
  double f_;
  double safe_;
  double behind_;
  std::string speed_mode_;
  // Setting (b) of docs/design.md section 8.6 adds a bounded acceleration term. Its bound
  // has to come from published data on human arm movement, and none has been
  // opened yet, so it is zero until one is (docs/design.md section 17.2).
  double accel_;
  // His speed is read from the parts of him above this height. His feet stand in
  // the floor's depth noise, flicker in and out of the picture and read as 0.4 to
  // 0.6 m/s while he stands still, and they cannot reach an arm that stays above
  // the worktop.
  double timed_above_;

  // the cells that hold him, or may: row, column, where along the axis he may be
  // (near to far), when, how fast that part of him moves, and whether it is only
  // remembered under the arm rather than seen
  std::vector<int> rows_;
  std::vector<int> cols_;
  std::vector<double> near_;
  std::vector<double> far_;
  std::vector<double> seen_at_;
  std::vector<double> cell_speed_;   // m/s, per cell of him
  std::vector<char> hidden_;

  Grid background_;                  // the empty cell's depth per grid cell
  bool has_background_ = false;
  GridB seen_;                       // this picture's cells of him
  Grid far_px_;                      // his farthest pixel per cell, for the speed
  bool has_picture_ = false;
  double taken_at_ = 0.0;
  double speed_ = 0.0;               // his fastest part, for logs
  double measured_ = 0.0;            // the same before holding, for logs
  int remembered_ = 0;               // cells kept under the arm's shadow

  std::deque<std::pair<double, Points>> history_;
  std::deque<std::pair<double, Grid>> fields_;
  VecX rx_;                          // ray slope per column
  VecX ry_;                          // and per row
  double half_diag_;                 // centre to corner of a cell, pixels
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_VIEWGRID_HPP
