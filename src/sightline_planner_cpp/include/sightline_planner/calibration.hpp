// Where the planner believes the eyes camera is, and what it believes it sees.
//
// A cell knows this from its own calibration, never perfectly. The error is a
// scene choice and is stated in docs/notes.md.
#ifndef SIGHTLINE_PLANNER_CALIBRATION_HPP
#define SIGHTLINE_PLANNER_CALIBRATION_HPP

#include "sightline_planner/types.hpp"

namespace sightline {

struct Calibration {
  Vec3 pos = Vec3::Zero();
  Mat3 R = Mat3::Identity();
  int width = 0;
  int height = 0;
  double fovy_deg = 0.0;

  double focal() const;

  // Unit direction of every pixel, in the world frame. Built once.
  Points rays() const;

  // Cosine between each pixel ray and the camera axis, to turn depth into range.
  VecX axis_cos() const;
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_CALIBRATION_HPP
