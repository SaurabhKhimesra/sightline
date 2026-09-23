// The small types every module here shares.
//
// The arm has six joints, so joint vectors are fixed at six and live on the stack.
// Sets of points in the cell are N by 3 and row major, which is how they come out of
// the camera and how they go into MuJoCo.
#ifndef SIGHTLINE_PLANNER_TYPES_HPP
#define SIGHTLINE_PLANNER_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace sightline {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;
using ArrX = Eigen::ArrayXd;
using ArrXX = Eigen::ArrayXXd;
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// Images and grids are row major and indexed (row, column), so flattening them
// gives the pixel order the camera sends, v * width + u.
using Grid = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using GridB = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using GridI = Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr int kNq = 6;

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_TYPES_HPP
