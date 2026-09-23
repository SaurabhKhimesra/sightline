// What the robot works out about the cell from its two cameras.
//
// docs/design.md section 8.3. Depth to points, background out, its own arm out, the box out,
// and whatever is left is treated as the person. Behind the person, the space the
// camera cannot see is treated as occupied too, which is the depth space idea of
// Flacco et al. 2012.
//
// Nothing here knows there is a person. It knows there is something that is not the
// station, not the robot and not the box, and that is enough: a loose part left on the
// bench counts as person, which is the conservative way round.
#ifndef SIGHTLINE_PLANNER_PERCEIVE_HPP
#define SIGHTLINE_PLANNER_PERCEIVE_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "sightline_planner/calibration.hpp"
#include "sightline_planner/frame.hpp"
#include "sightline_planner/kin.hpp"
#include "sightline_planner/types.hpp"

namespace sightline {

constexpr double kVoxelM = 0.02;            // person voxel size, docs/design.md section 8.3 (scene choice)
constexpr double kUnseenDepthM = 0.30;      // how far behind a person point counts as occupied (scene choice)
constexpr double kBackgroundMarginM = 0.04; // least a point must beat the empty station by (scene choice)
constexpr double kBackgroundSigmas = 3.0;   // and at least this many standard deviations of depth noise
constexpr double kRobotMarginM = 0.05;      // points within this of the robot's own shape are its own (scene choice)
constexpr int kMinPointsPerVoxel = 2;       // one stray point is noise (scene choice)
constexpr double kJigYawDeg = 10.0;         // how far out of square the jig can hold a part (scene choice)
// The arm and the parts are grown by this many pixels before subtracting (scene choice).
// Two was not enough: the lateral noise hands a pixel 3 to 4 px outside the box's edge
// the box's own depth, 100 mm nearer than the worktop behind it
constexpr int kSilhouettePx = 4;
constexpr int kLateralReachPx = 3;          // how far sideways the camera's lateral noise can fetch a sample from (scene choice)
// A return this far in front of the robot's own rendered surface is still the robot: at
// a link's edge the camera's lateral noise samples the link's front face, up to a link
// radius (60 mm) nearer than the edge, and the depth noise adds 12 mm at 3 m. Two such
// pixels at the tool's edge, 70 mm in front of the render, read as a hand touching the
// arm and sent it backing away from itself (scene choice)
constexpr double kRobotDepthTolM = 0.08;

// A depth image of what the planner already knows is there: its own arm at its
// current joint angles and the parts at their fitted pose, seen from where it
// believes the eyes camera is. Anything clearly in front of it, and of the empty
// cell, is the person. Rendered offscreen through EGL.
class KnownWorld {
 public:
  static constexpr const char *kParts[3] = {"wp_base", "wp_rail", "wp_cover"};

  KnownWorld(mjModel *model, const Calibration &calib);
  ~KnownWorld();

  KnownWorld(const KnownWorld &) = delete;
  KnownWorld &operator=(const KnownWorld &) = delete;

  // parts maps a part name to its origin in the cell. A part left out is not there.
  Grid depth(const Vec6 &q, const std::map<std::string, Vec3> &parts);

  void close();

 private:
  mjModel *m_;
  mjData *d_ = nullptr;
  Calibration calib_;
  mjvOption opt_{};
  mjvCamera cam_{};
  mjvScene scn_{};
  mjrContext con_{};
  std::map<std::string, int> mocap_;
  bool open_ = false;
  void *egl_display_ = nullptr;
  void *egl_context_ = nullptr;
};

// Everything the planner thinks is in the way, at one moment.
struct BoxPose {
  double x = 0.0;
  double y = 0.0;
  double top_z = 0.0;
  double yaw = 0.0;
};

struct PersonModel {
  double t = 0.0;
  Points voxels = Points(0, 3);        // centres, m
  Points velocity = Points(0, 3);      // m/s
  Points unseen = Points(0, 3);        // behind the person
  int points = 0;
  std::optional<BoxPose> box_pose;     // this frame
  std::optional<BoxPose> box_pose_held;  // the last one that passed, which the planner uses
  double box_age_s = 0.0;
  double speed_max = 0.0;
  std::vector<int> person_px;          // pixel index
  std::vector<double> person_z;        // depth along the axis, m
};

// The eyes pipeline, one frame at a time.
class Perception {
 public:
  Perception(const Calibration &calib, ToolKinematics *kin, const Taught &taught,
             double voxel = kVoxelM, double unseen_depth = kUnseenDepthM,
             KnownWorld *known = nullptr);

  // A depth map of the empty station, taken once at the start.
  void set_background(const Grid &depth);

  // Where each part the planner believes is in the jig sits, from the last box
  // fit if there is one and the jig's nominal pose if not. Height from the jig.
  std::map<std::string, Vec3> part_origins(const std::vector<std::string> &present) const;

  // parts_present names the parts the planner's own task state says are in the jig,
  // and q_at_capture is where the arm was when the picture was taken.
  PersonModel update(const Grid &depth, const Vec6 &q, double t,
                     const std::vector<std::string> &parts_present = {},
                     const Vec6 *q_at_capture = nullptr);

  const std::optional<BoxPose> &last_box() const { return last_box_; }

 private:
  struct RobotShape {
    int geom = 0;
    int type = 0;
    Vec3 size = Vec3::Zero();
  };

  // The robot's own collision shapes, from its own model, for self filtering.
  std::vector<RobotShape> robot_shapes() const;

  // Points inside the taught jig volume are the workpiece. Fit its top face.
  std::optional<BoxPose> fit_box(const Points &pts, std::vector<char> *inside) const;

  // True for points that are the robot's own arm or tool. A real cell does this from
  // its URDF and joint angles; here the model the planner plans with says where its
  // shapes are.
  std::vector<char> is_robot(const Points &pts, const Vec6 &q) const;

  void voxelize(const Points &pts, std::vector<int64_t> *keys, Points *centres) const;

  // Velocity from the nearest voxel of the frame before, within one cell.
  void track(const std::vector<int64_t> &keys, const Points &centres, double t, Points *velocity,
             double *speed_max);

  // Voxels behind the person along the camera ray, which the camera cannot see. They
  // count as occupied for R1: a hand could be there and the camera would not know.
  // Flacco et al. 2012 call this depth space.
  Points unseen_behind(const Points &centres) const;

  Calibration calib_;
  ToolKinematics *kin_;
  Taught taught_;
  KnownWorld *known_;
  VecX surface_;                 // nearest known surface per pixel, this frame
  bool has_surface_ = false;
  double voxel_;
  double unseen_depth_;
  VecX background_;
  VecX background_gate_;
  bool has_background_ = false;
  Points rays_;
  VecX range_scale_;
  std::vector<int64_t> prev_keys_;
  bool has_prev_ = false;
  double prev_t_ = 0.0;
  std::optional<BoxPose> last_box_;
  std::optional<double> last_box_t_;
  std::vector<RobotShape> robot_shapes_;
};

// Each pixel takes the nearest depth within px pixels: near objects grow sideways, to
// cover the depth camera's lateral noise. The price is a rim that wide around the arm
// where a person touching it at the same depth is explained away too, about 1.5 cm at
// 3 m against a 10 cm safety distance.
Grid grow_near(const Grid &depth, int px);

// Three voxel indices into one sortable integer, so lookups can be a binary search.
int64_t pack(int x, int y, int z);
void unpack(int64_t key, int *x, int *y, int *z);

// Does anything the planner thinks is in the way sit on this line of sight?
bool blocks_line(const Points &voxels, const Vec3 &start, const Vec3 &end, double radius);

// Where a hole was found in an image, and how big the blob was.
struct HoleBlob {
  double u = 0.0;
  double v = 0.0;
  int area = 0;
};

// The dark blob that is the hole, in image coordinates, or nothing: the connected blob
// nearest the prediction, checked against the hole's own diameter at the distance the
// tool is at. A plain centroid of everything dark in the window lands 7 to 16 mm off,
// because the shadow under the tool and the neighbouring hole are dark too.
std::optional<HoleBlob> find_hole(const Image &image, double predicted_u, double predicted_v,
                                  double expect_px, int window = 40, double near_px = 18.0);

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_PERCEIVE_HPP
