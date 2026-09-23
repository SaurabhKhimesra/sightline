#include "sightline_planner/viewgrid.hpp"

#include <algorithm>
#include <cmath>

namespace sightline {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Square max filter of half width r, done as two passes of shifted maxima.
Grid max_filter(const Grid &a, int r) {
  if (r <= 0) {
    return a;
  }
  const int h = static_cast<int>(a.rows());
  const int w = static_cast<int>(a.cols());
  Grid rows_done = a;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      double best = a(i, j);
      for (int k = std::max(i - r, 0); k <= std::min(i + r, h - 1); ++k) {
        best = std::max(best, a(k, j));
      }
      rows_done(i, j) = best;
    }
  }
  Grid out = rows_done;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      double best = rows_done(i, j);
      for (int k = std::max(j - r, 0); k <= std::min(j + r, w - 1); ++k) {
        best = std::max(best, rows_done(i, k));
      }
      out(i, j) = best;
    }
  }
  return out;
}

// How many cells are set in the square of half width r around each cell.
GridI window_count(const GridB &mask, int r) {
  const int h = static_cast<int>(mask.rows());
  const int w = static_cast<int>(mask.cols());
  GridI sums = GridI::Zero(h + 1, w + 1);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      sums(i + 1, j + 1) = (mask(i, j) ? 1 : 0) + sums(i, j + 1) + sums(i + 1, j) - sums(i, j);
    }
  }
  GridI out(h, w);
  for (int i = 0; i < h; ++i) {
    const int i0 = std::max(i - r, 0);
    const int i1 = std::min(i + r, h - 1);
    for (int j = 0; j < w; ++j) {
      const int j0 = std::max(j - r, 0);
      const int j1 = std::min(j + r, w - 1);
      out(i, j) = sums(i1 + 1, j1 + 1) - sums(i0, j1 + 1) - sums(i1 + 1, j0) + sums(i0, j0);
    }
  }
  return out;
}

}  // namespace

ViewGrid::ViewGrid(const Calibration &calib, int cell_px, double safe, double behind,
                   const std::string &speed_mode, double accel, double timed_above)
    : calib_(calib),
      cell_(cell_px),
      gw_(calib.width / cell_px),
      gh_(calib.height / cell_px),
      f_(calib.focal()),
      safe_(safe),
      behind_(behind),
      speed_mode_(speed_mode),
      accel_(accel),
      timed_above_(timed_above) {
  seen_ = GridB::Constant(gh_, gw_, false);
  far_px_ = Grid::Constant(gh_, gw_, -kInf);
  speed_ = speed_mode_ == "iso" ? kIsoSpeed : 0.0;
  rx_.resize(gw_);
  ry_.resize(gh_);
  for (int c = 0; c < gw_; ++c) {
    rx_[c] = ((c + 0.5) * cell_ - calib_.width / 2.0) / f_;
  }
  for (int r = 0; r < gh_; ++r) {
    ry_[r] = -((r + 0.5) * cell_ - calib_.height / 2.0) / f_;
  }
  half_diag_ = cell_ / std::sqrt(2.0);
}

double ViewGrid::reach(double age, double speed) const {
  const double a = std::max(age, 0.0);
  return speed * a + 0.5 * accel_ * a * a;
}

void ViewGrid::set_background(const Grid &depth) {
  background_ = Grid::Constant(gh_, gw_, -kInf);
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      double worst = -kInf;
      for (int dv = 0; dv < cell_; ++dv) {
        for (int du = 0; du < cell_; ++du) {
          const double z = depth(r * cell_ + dv, c * cell_ + du);
          worst = std::max(worst, z > 0 ? z : kInf);
        }
      }
      background_(r, c) = worst;
    }
  }
  has_background_ = true;
}

Points ViewGrid::points_at(const std::vector<int> &rows, const std::vector<int> &cols,
                           const std::vector<double> &z) const {
  Points out(static_cast<Eigen::Index>(rows.size()), 3);
  for (size_t i = 0; i < rows.size(); ++i) {
    out(static_cast<Eigen::Index>(i), 0) = rx_[cols[i]] * z[i];
    out(static_cast<Eigen::Index>(i), 1) = ry_[rows[i]] * z[i];
    out(static_cast<Eigen::Index>(i), 2) = -z[i];
  }
  return out;
}

void ViewGrid::update(const std::vector<int> &person_px, const std::vector<double> &person_z,
                      double taken_at, const Grid *shadow) {
  const int n_cells = gh_ * gw_;
  std::vector<double> near_flat(n_cells, kInf);
  std::vector<double> far_flat(n_cells, -kInf);
  std::vector<int> count(n_cells, 0);
  for (size_t i = 0; i < person_px.size(); ++i) {
    const int u = person_px[i] % calib_.width;
    const int v = person_px[i] / calib_.width;
    const int k = (v / cell_) * gw_ + (u / cell_);
    near_flat[k] = std::min(near_flat[k], person_z[i]);
    far_flat[k] = std::max(far_flat[k], person_z[i]);
    count[k] += 1;
  }
  for (int k = 0; k < n_cells; ++k) {
    if (count[k] < kMinPixels) {
      near_flat[k] = kInf;
    }
  }
  Grid near_g(gh_, gw_);
  far_px_ = Grid(gh_, gw_);
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      near_g(r, c) = near_flat[r * gw_ + c];
      far_px_(r, c) = far_flat[r * gw_ + c];
    }
  }
  GridB seen(gh_, gw_);
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      seen(r, c) = std::isfinite(near_g(r, c));
    }
  }
  const GridI neighbours = window_count(seen, 1);
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      seen(r, c) = seen(r, c) && neighbours(r, c) > 1;   // itself and at least one neighbour
      if (!seen(r, c)) {
        near_g(r, c) = kInf;
      }
    }
  }

  const Grid field = measure_speed(seen, near_g, taken_at);

  std::vector<int> rows;
  std::vector<int> cols;
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      if (seen(r, c)) {
        rows.push_back(r);
        cols.push_back(c);
      }
    }
  }
  std::vector<double> near(rows.size());
  std::vector<double> far(rows.size());
  std::vector<double> speed(rows.size());
  std::vector<double> seen_at(rows.size(), taken_at);
  std::vector<char> hidden(rows.size(), 0);
  for (size_t i = 0; i < rows.size(); ++i) {
    near[i] = near_g(rows[i], cols[i]);
    far[i] = near[i] + behind_;
    speed[i] = field(rows[i], cols[i]);
  }

  remembered_ = 0;
  if (shadow != nullptr && !rows_.empty()) {
    // what the arm covers now keeps what was there, with the time it was seen
    for (size_t i = 0; i < rows_.size(); ++i) {
      const bool under_arm = std::isfinite((*shadow)(rows_[i], cols_[i]));
      if (!under_arm || seen(rows_[i], cols_[i]) || taken_at - seen_at_[i] > kMemoryS) {
        continue;
      }
      rows.push_back(rows_[i]);
      cols.push_back(cols_[i]);
      near.push_back(near_[i]);
      far.push_back(far_[i]);
      speed.push_back(std::min(cell_speed_[i], kIsoSpeed));
      seen_at.push_back(seen_at_[i]);
      hidden.push_back(1);
      remembered_ += 1;
    }
  }

  rows_ = std::move(rows);
  cols_ = std::move(cols);
  near_ = std::move(near);
  far_ = std::move(far);
  seen_at_ = std::move(seen_at);
  cell_speed_ = std::move(speed);
  hidden_ = std::move(hidden);
  seen_ = seen;
  taken_at_ = taken_at;
  has_picture_ = true;
}

Grid ViewGrid::measure_speed(const GridB &seen, const Grid &near, double t) {
  // only where a real piece of him is: two noise cells side by side, which the
  // neighbour test lets through, would otherwise read as a jump of a metre
  const GridI solid = window_count(seen, 2);
  std::vector<int> vs;
  std::vector<int> us;
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      if (seen(r, c) && solid(r, c) >= kSolidCells) {
        vs.push_back(r);
        us.push_back(c);
      }
    }
  }
  std::vector<double> near_at(vs.size());
  std::vector<double> far_at(vs.size());
  for (size_t i = 0; i < vs.size(); ++i) {
    near_at[i] = near(vs[i], us[i]);
    far_at[i] = far_px_(vs[i], us[i]);
  }
  Points pts = points_at(vs, us, near_at);
  Points back = points_at(vs, us, far_at);
  if (std::isfinite(timed_above_) && pts.rows() > 0) {
    std::vector<int> keep_v;
    std::vector<int> keep_u;
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < pts.rows(); ++i) {
      const Vec3 world = calib_.R * pts.row(i).transpose() + calib_.pos;
      if (world[2] > timed_above_) {
        keep.push_back(i);
        keep_v.push_back(vs[static_cast<size_t>(i)]);
        keep_u.push_back(us[static_cast<size_t>(i)]);
      }
    }
    Points kept_pts(static_cast<Eigen::Index>(keep.size()), 3);
    Points kept_back(static_cast<Eigen::Index>(keep.size()), 3);
    for (size_t i = 0; i < keep.size(); ++i) {
      kept_pts.row(static_cast<Eigen::Index>(i)) = pts.row(keep[i]);
      kept_back.row(static_cast<Eigen::Index>(i)) = back.row(keep[i]);
    }
    vs = keep_v;
    us = keep_u;
    pts = kept_pts;
    back = kept_back;
  }

  // this picture's nearest pixels, compared later with both layers of this one
  Points both(pts.rows() + back.rows(), 3);
  both.topRows(pts.rows()) = pts;
  both.bottomRows(back.rows()) = back;
  history_.emplace_back(t, both);
  while (static_cast<int>(history_.size()) > kSpeedBaseline + 1) {
    history_.pop_front();
  }

  Grid local = Grid::Zero(gh_, gw_);
  double raw = 0.0;
  if (static_cast<int>(history_.size()) > kSpeedBaseline && pts.rows() > 0) {
    const double t_old = history_.front().first;
    const Points &old = history_.front().second;
    if (old.rows() > 0 && t > t_old) {
      std::vector<double> speeds(static_cast<size_t>(pts.rows()));
      for (Eigen::Index i = 0; i < pts.rows(); ++i) {
        double gap = kInf;
        for (Eigen::Index j = 0; j < old.rows(); ++j) {
          gap = std::min(gap, (pts.row(i) - old.row(j)).squaredNorm());
        }
        const double speed = std::min(std::sqrt(gap) / (t - t_old), kSpeedCap);
        speeds[static_cast<size_t>(i)] = speed;
        local(vs[static_cast<size_t>(i)], us[static_cast<size_t>(i)]) = speed;
      }
      std::vector<double> ranked = speeds;
      const size_t k = std::min<size_t>(kSpeedRank, ranked.size());
      std::nth_element(ranked.begin(), ranked.begin() + (ranked.size() - k), ranked.end());
      raw = ranked[ranked.size() - k];
    }
  }

  // a reading needs a neighbour that agrees: the second largest of its 3 by 3
  Grid agreed = local;
  for (int r = 0; r < gh_; ++r) {
    for (int c = 0; c < gw_; ++c) {
      double best = 0.0;
      double second = 0.0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int rr = r + dy;
          const int cc = c + dx;
          const double value = (rr < 0 || rr >= gh_ || cc < 0 || cc >= gw_) ? 0.0 : local(rr, cc);
          if (value > best) {
            second = best;
            best = value;
          } else if (value > second) {
            second = value;
          }
        }
      }
      agreed(r, c) = std::min(local(r, c), second);
    }
  }
  local = agreed;

  // the leading edge's speed for the whole part, a hand's size round it
  double z = 3.0;
  if (!vs.empty()) {
    std::vector<double> depths = near_at;
    depths.resize(vs.size());
    for (size_t i = 0; i < vs.size(); ++i) {
      depths[i] = near(vs[i], us[i]);
    }
    std::vector<double> sorted = depths;
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    z = sorted.size() % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
  }
  Grid field = max_filter(local, static_cast<int>(std::ceil(kPartM * f_ / (z * cell_))));
  fields_.emplace_back(t, field);
  while (!fields_.empty() && fields_.front().first < t - kSpeedHoldS) {
    fields_.pop_front();
  }
  Grid held = fields_.front().second;
  for (const auto &entry : fields_) {
    held = held.max(entry.second);
  }
  measured_ = raw;
  if (speed_mode_ == "iso") {
    return Grid::Constant(gh_, gw_, kIsoSpeed);
  }
  speed_ = held.maxCoeff();
  return held;
}

Points ViewGrid::seen_points() const {
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<double> near;
  for (size_t i = 0; i < rows_.size(); ++i) {
    if (!hidden_[i]) {
      rows.push_back(rows_[i]);
      cols.push_back(cols_[i]);
      near.push_back(near_[i]);
    }
  }
  Points local = points_at(rows, cols, near);
  Points out(local.rows(), 3);
  for (Eigen::Index i = 0; i < local.rows(); ++i) {
    out.row(i) = (calib_.R * local.row(i).transpose() + calib_.pos).transpose();
  }
  return out;
}

Vec3 ViewGrid::cell_point(int k) const {
  const Points local = points_at({rows_[k]}, {cols_[k]}, {near_[k]});
  return calib_.R * local.row(0).transpose() + calib_.pos;
}

double ViewGrid::speed_near(const Vec3 &point, double within) const {
  if (rows_.empty()) {
    return 0.0;
  }
  const Points local = points_at(rows_, cols_, near_);
  double best = -kInf;
  bool any = false;
  for (Eigen::Index i = 0; i < local.rows(); ++i) {
    if (hidden_[static_cast<size_t>(i)]) {
      continue;
    }
    const Vec3 world = calib_.R * local.row(i).transpose() + calib_.pos;
    if ((world - point).norm() < within) {
      best = std::max(best, cell_speed_[static_cast<size_t>(i)]);
      any = true;
    }
  }
  return any ? best : std::nan("");
}

void ViewGrid::project(const Points &points, VecX *u, VecX *v, VecX *depth,
                       std::vector<char> *ahead) const {
  const Eigen::Index n = points.rows();
  u->resize(n);
  v->resize(n);
  depth->resize(n);
  ahead->assign(static_cast<size_t>(n), 0);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Vec3 rel = calib_.R.transpose() * (points.row(i).transpose() - calib_.pos);
    const double z = -rel[2];
    (*depth)[i] = z;
    const bool in_front = z > 1e-3;
    (*ahead)[static_cast<size_t>(i)] = in_front ? 1 : 0;
    const double safe_depth = in_front ? z : 1.0;
    (*u)[i] = calib_.width / 2.0 + f_ * rel[0] / safe_depth;
    (*v)[i] = calib_.height / 2.0 - f_ * rel[1] / safe_depth;
  }
}

Excess ViewGrid::excess(const Points &points, const VecX &radii, double now,
                        bool want_cells) const {
  return excess(points, radii, VecX::Constant(points.rows(), now), want_cells);
}

// Two parts, added when both bite, so that every way toward them reads as worse:
// R2 is in front of a cell where they are seen with the line of sight closer than the
// margin, R1 is closer than the margin to where they may be along that line of sight.
// Sideways alone, moving down the line of sight onto a hand would not change the
// number at all.
Excess ViewGrid::excess(const Points &points, const VecX &radii, const VecX &now,
                        bool want_cells) const {
  const Eigen::Index n = points.rows();
  Excess out;
  out.depth = VecX::Constant(n, -kInf);
  if (want_cells) {
    out.cells.assign(static_cast<size_t>(n), -1);
  }
  if (rows_.empty() || n == 0) {
    return out;
  }
  VecX u;
  VecX v;
  VecX depth;
  std::vector<char> ahead;
  project(points, &u, &v, &depth, &ahead);
  std::vector<Eigen::Index> idx;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (ahead[static_cast<size_t>(i)]) {
      idx.push_back(i);
    }
  }
  if (idx.empty()) {
    return out;
  }

  double when_max = -kInf;
  double radius_max = -kInf;
  double depth_min = kInf;
  double u_lo = kInf;
  double u_hi = -kInf;
  double v_lo = kInf;
  double v_hi = -kInf;
  for (Eigen::Index i : idx) {
    when_max = std::max(when_max, now[i]);
    radius_max = std::max(radius_max, radii[i]);
    depth_min = std::min(depth_min, depth[i]);
    u_lo = std::min(u_lo, u[i]);
    u_hi = std::max(u_hi, u[i]);
    v_lo = std::min(v_lo, v[i]);
    v_hi = std::max(v_hi, v[i]);
  }
  // only cells that could be within the largest margin of some point matter
  double widest = -kInf;
  for (size_t j = 0; j < rows_.size(); ++j) {
    widest = std::max(widest, reach(when_max - seen_at_[j], cell_speed_[j]));
  }
  const double reach_px = (safe_ + radius_max + widest) * f_ / depth_min + half_diag_;

  std::vector<size_t> box;
  for (size_t j = 0; j < rows_.size(); ++j) {
    const double cu = (cols_[j] + 0.5) * cell_;
    const double cv = (rows_[j] + 0.5) * cell_;
    if (cu > u_lo - reach_px && cu < u_hi + reach_px && cv > v_lo - reach_px &&
        cv < v_hi + reach_px) {
      box.push_back(j);
    }
  }
  if (box.empty()) {
    return out;
  }

  for (Eigen::Index i : idx) {
    const double z = depth[i];
    double best = -kInf;
    size_t best_cell = box.front();
    for (size_t j : box) {
      const double cu = (cols_[j] + 0.5) * cell_;
      const double cv = (rows_[j] + 0.5) * cell_;
      const double grow = reach(now[i] - seen_at_[j], cell_speed_[j]);
      const double margin = safe_ + radii[i] + grow;
      // the point's line of sight to the nearest corner of the cell, in metres at its depth
      const double px = std::hypot(u[i] - cu, v[i] - cv) - half_diag_;
      const double side = std::max(px, 0.0) * z / f_;
      const double along = std::max(near_[j] - z, 0.0) + std::max(z - far_[j], 0.0);
      const double r1 = margin - std::sqrt(side * side + along * along);
      const double r2 = (z < near_[j] && !hidden_[j]) ? margin - side : -kInf;
      const double both = (r1 > 0 || r2 > 0) ? std::max(r1, 0.0) + std::max(r2, 0.0)
                                             : std::max(r1, r2);
      if (both > best) {
        best = both;
        best_cell = j;
      }
    }
    out.depth[i] = best;
    if (want_cells) {
      out.cells[static_cast<size_t>(i)] = static_cast<int>(best_cell);
    }
  }
  return out;
}

std::vector<char> ViewGrid::forbidden(const Points &points, const VecX &radii,
                                      double now) const {
  const Excess e = excess(points, radii, now);
  std::vector<char> out(static_cast<size_t>(points.rows()), 0);
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    out[static_cast<size_t>(i)] = e.depth[i] > 0 ? 1 : 0;
  }
  return out;
}

Grid ViewGrid::shadow_of(const Points &points, const VecX &radii) const {
  Grid out = Grid::Constant(gh_, gw_, kInf);
  VecX u;
  VecX v;
  VecX depth;
  std::vector<char> ahead;
  project(points, &u, &v, &depth, &ahead);
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    if (!ahead[static_cast<size_t>(i)]) {
      continue;
    }
    const double r = radii[i] * f_ / depth[i];                 // pixels
    int c0 = static_cast<int>(std::floor((u[i] - r) / cell_));
    int c1 = static_cast<int>(std::floor((u[i] + r) / cell_));
    int r0 = static_cast<int>(std::floor((v[i] - r) / cell_));
    int r1 = static_cast<int>(std::floor((v[i] + r) / cell_));
    if (c1 < 0 || r1 < 0 || c0 >= gw_ || r0 >= gh_) {
      continue;
    }
    c0 = std::max(c0, 0);
    r0 = std::max(r0, 0);
    c1 = std::min(c1, gw_ - 1);
    r1 = std::min(r1, gh_ - 1);
    const double front = depth[i] - radii[i];
    for (int rr = r0; rr <= r1; ++rr) {
      for (int cc = c0; cc <= c1; ++cc) {
        out(rr, cc) = std::min(out(rr, cc), front);
      }
    }
  }
  return out;
}

}  // namespace sightline
