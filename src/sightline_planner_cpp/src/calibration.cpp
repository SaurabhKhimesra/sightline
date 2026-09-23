#include "sightline_planner/calibration.hpp"

#include <cmath>

namespace sightline {

double Calibration::focal() const {
  return (height / 2.0) / std::tan(fovy_deg * M_PI / 180.0 / 2.0);
}

Points Calibration::rays() const {
  const double f = focal();
  Points out(static_cast<Eigen::Index>(width) * height, 3);
  for (int v = 0; v < height; ++v) {
    const double dy = -(v + 0.5 - height / 2.0) / f;
    for (int u = 0; u < width; ++u) {
      const double dx = (u + 0.5 - width / 2.0) / f;
      Vec3 dir(dx, dy, -1.0);
      dir.normalize();
      out.row(static_cast<Eigen::Index>(v) * width + u) = (R * dir).transpose();
    }
  }
  return out;
}

VecX Calibration::axis_cos() const {
  const double f = focal();
  VecX out(static_cast<Eigen::Index>(width) * height);
  for (int v = 0; v < height; ++v) {
    const double dy = (v + 0.5 - height / 2.0) / f;
    for (int u = 0; u < width; ++u) {
      const double dx = (u + 0.5 - width / 2.0) / f;
      out[static_cast<Eigen::Index>(v) * width + u] = 1.0 / std::sqrt(1.0 + dx * dx + dy * dy);
    }
  }
  return out;
}

}  // namespace sightline
