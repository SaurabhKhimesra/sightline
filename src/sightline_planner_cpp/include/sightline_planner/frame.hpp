// What the planner is allowed to see, and nothing else.
//
// docs/design.md section 8.1: eyes camera colour and depth, wrist camera colour, its own
// joint positions and velocities, the static station model, its own robot model, the
// product model (holes relative to the part), and the task state it keeps itself.
//
// It never gets human geometry, segmentation, the true box pose, or the future.
#ifndef SIGHTLINE_PLANNER_FRAME_HPP
#define SIGHTLINE_PLANNER_FRAME_HPP

#include <string>
#include <vector>

#include "sightline_planner/types.hpp"

namespace sightline {

// One picture from a camera, as the planner gets it: row major, top row first.
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> data;

  bool empty() const { return data.empty(); }
  float at(int v, int u, int c = 0) const {
    return data[static_cast<size_t>((v * width + u) * channels + c)];
  }
};

// One tick of what the cell tells the robot.
struct SensorFrame {
  double t = 0.0;
  Vec6 q = Vec6::Zero();    // joint positions, rad
  Vec6 qd = Vec6::Zero();   // joint velocities, rad/s
  Image wrist_rgb;          // 640 x 480 colour, 25 Hz
  Image eyes_rgb;           // 640 x 480 colour, 25 Hz, one frame late
  Image eyes_depth;         // metres, same frame
  // what a jig switch or a button tells the cell. Not a look at the person.
  // empty, rail_ready, cover_on, cover_ready, done: what the jig switches say
  std::string jig_signal = "empty";
};

// Static cell knowledge: where the jig holds the part and where the screws are.
//
// The jig is bolted to the bench, so its nominal part pose belongs to the station
// model. Hole positions are given relative to the part, as docs/design.md section 8.1
// allows. Any difference between the nominal pose and the real one is what the
// wrist camera is for.
struct Taught {
  Vec3 part_pose = Vec3::Zero();        // x, y, z of the part's origin in the cell
  std::vector<Vec3> rail_holes;         // relative to the part origin
  std::vector<Vec3> cover_holes;
  Vec3 feeder_pick = Vec3::Zero();
  double screw_length = 0.0156;
  double approach = 0.06;               // stand-off above a hole, m (scene choice)
  double cover_size_x = 0.200;          // the part's own drawing, not its pose
  double cover_size_y = 0.150;
  double cover_top = 0.086;             // top face above the part origin, from the drawing
  double rail_origin_dz = 0.021;        // the rail sits on its bosses, from the drawing
  double cover_origin_dz = 0.082;       // the cover's own origin, 4 mm under its top face

  Vec3 world(const Vec3 &hole) const { return part_pose + hole; }
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_FRAME_HPP
