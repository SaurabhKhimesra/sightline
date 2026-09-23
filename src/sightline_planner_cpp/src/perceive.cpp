#include "sightline_planner/perceive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <EGL/egl.h>
#include <EGL/eglext.h>

namespace sightline {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The camera's own depth noise at this range, from Nguyen et al. 2012 as the cell
// applies it. The same curve gates the background and the known world.
double depth_sigma(double range) {
  const double clipped = std::min(std::max(range, 0.4), 2.75);
  return 0.0012 + 0.0019 * (clipped - 0.4) * (clipped - 0.4);
}

// Square minimum filter of half width r, as two passes of shifted minima.
Grid min_filter(const Grid &a, int r) {
  const int h = static_cast<int>(a.rows());
  const int w = static_cast<int>(a.cols());
  Grid rows_done(h, w);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      double best = a(i, j);
      for (int k = std::max(i - r, 0); k <= std::min(i + r, h - 1); ++k) {
        best = std::min(best, a(k, j));
      }
      rows_done(i, j) = best;
    }
  }
  Grid out(h, w);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      double best = rows_done(i, j);
      for (int k = std::max(j - r, 0); k <= std::min(j + r, w - 1); ++k) {
        best = std::min(best, rows_done(i, k));
      }
      out(i, j) = best;
    }
  }
  return out;
}

// numpy's linear percentile, so the box fit lands where the Python one does.
double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const double pos = p / 100.0 * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return values[lo];
  }
  return values[lo] + (pos - lo) * (values[hi] - values[lo]);
}

double median_of(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

}  // namespace

// ---------------------------------------------------------------- voxel keys

// voxel indices run from -2^20 to 2^20, which is +-20 km at 2 cm
constexpr int64_t kShift = 1LL << 20;
constexpr int64_t kMask = (1LL << 21) - 1;

int64_t pack(int x, int y, int z) {
  return ((static_cast<int64_t>(x) + kShift) << 42) |
         ((static_cast<int64_t>(y) + kShift) << 21) | (static_cast<int64_t>(z) + kShift);
}

void unpack(int64_t key, int *x, int *y, int *z) {
  *x = static_cast<int>((key >> 42) - kShift);
  *y = static_cast<int>(((key >> 21) & kMask) - kShift);
  *z = static_cast<int>((key & kMask) - kShift);
}

Grid grow_near(const Grid &depth, int px) {
  if (px <= 0) {
    return depth;
  }
  const int h = static_cast<int>(depth.rows());
  const int w = static_cast<int>(depth.cols());
  Grid out = depth;
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      double best = depth(v, u);
      for (int dv = -px; dv <= px; ++dv) {
        const int vv = v + dv;
        if (vv < 0 || vv >= h) {
          continue;
        }
        for (int du = -px; du <= px; ++du) {
          const int uu = u + du;
          if (uu < 0 || uu >= w) {
            continue;
          }
          best = std::min(best, depth(vv, uu));
        }
      }
      out(v, u) = best;
    }
  }
  return out;
}

bool blocks_line(const Points &voxels, const Vec3 &start, const Vec3 &end, double radius) {
  if (voxels.rows() == 0) {
    return false;
  }
  const Vec3 seg = end - start;
  const double length = seg.norm();
  if (length < 1e-9) {
    return false;
  }
  const Vec3 direction = seg / length;
  for (Eigen::Index i = 0; i < voxels.rows(); ++i) {
    const Vec3 rel = voxels.row(i).transpose() - start;
    const double along = std::min(std::max(rel.dot(direction), 0.0), length);
    if ((rel - along * direction).norm() < radius) {
      return true;
    }
  }
  return false;
}

namespace {

// Connected components of a small boolean image, four way.
std::vector<int> label_blobs(const std::vector<char> &mask, int h, int w, int *count) {
  std::vector<int> labels(mask.size(), 0);
  int tag = 0;
  std::vector<std::pair<int, int>> stack;
  for (int sv = 0; sv < h; ++sv) {
    for (int su = 0; su < w; ++su) {
      if (!mask[static_cast<size_t>(sv * w + su)] || labels[static_cast<size_t>(sv * w + su)]) {
        continue;
      }
      tag += 1;
      stack.emplace_back(sv, su);
      labels[static_cast<size_t>(sv * w + su)] = tag;
      while (!stack.empty()) {
        const auto [cv, cu] = stack.back();
        stack.pop_back();
        static const int dv[4] = {1, -1, 0, 0};
        static const int du[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
          const int nv = cv + dv[k];
          const int nu = cu + du[k];
          if (nv < 0 || nv >= h || nu < 0 || nu >= w) {
            continue;
          }
          const size_t at = static_cast<size_t>(nv * w + nu);
          if (mask[at] && !labels[at]) {
            labels[at] = tag;
            stack.emplace_back(nv, nu);
          }
        }
      }
    }
  }
  *count = tag;
  return labels;
}

}  // namespace

std::optional<HoleBlob> find_hole(const Image &image, double predicted_u, double predicted_v,
                                  double expect_px, int window, double near_px) {
  const int h = image.height;
  const int w = image.width;
  const int u0 = static_cast<int>(std::max(predicted_u - window, 0.0));
  const int u1 = static_cast<int>(std::min(predicted_u + window, static_cast<double>(w)));
  const int v0 = static_cast<int>(std::max(predicted_v - window, 0.0));
  const int v1 = static_cast<int>(std::min(predicted_v + window, static_cast<double>(h)));
  if (u1 - u0 < 12 || v1 - v0 < 12) {
    return std::nullopt;
  }
  const int ww = u1 - u0;
  const int wh = v1 - v0;
  std::vector<double> win(static_cast<size_t>(ww) * wh);
  for (int v = 0; v < wh; ++v) {
    for (int u = 0; u < ww; ++u) {
      double sum = 0.0;
      for (int c = 0; c < image.channels; ++c) {
        sum += image.at(v0 + v, u0 + u, c);
      }
      win[static_cast<size_t>(v * ww + u)] = sum / std::max(image.channels, 1);
    }
  }
  const double cut = 0.55 * median_of(win);
  std::vector<char> dark(win.size());
  bool any = false;
  for (size_t i = 0; i < win.size(); ++i) {
    dark[i] = win[i] < cut ? 1 : 0;
    any = any || dark[i];
  }
  if (!any) {
    return std::nullopt;
  }
  int tags = 0;
  const std::vector<int> labels = label_blobs(dark, wh, ww, &tags);
  const double want_area = M_PI * (expect_px / 2) * (expect_px / 2);
  std::optional<HoleBlob> best;
  double best_score = kInf;
  for (int tag = 1; tag <= tags; ++tag) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (int v = 0; v < wh; ++v) {
      for (int u = 0; u < ww; ++u) {
        if (labels[static_cast<size_t>(v * ww + u)] == tag) {
          xs.push_back(u);
          ys.push_back(v);
        }
      }
    }
    const int area = static_cast<int>(xs.size());
    if (area < std::max(6.0, 0.25 * want_area) || area > 4.0 * want_area) {
      continue;
    }
    const double mean_x = std::accumulate(xs.begin(), xs.end(), 0.0) / area;
    const double mean_y = std::accumulate(ys.begin(), ys.end(), 0.0) / area;
    const double cu = u0 + mean_x;
    const double cv = v0 + mean_y;
    const double off = std::hypot(cu - predicted_u, cv - predicted_v);
    if (off > near_px) {
      continue;
    }
    double var_x = 0.0;
    double var_y = 0.0;
    for (int i = 0; i < area; ++i) {
      var_x += (xs[static_cast<size_t>(i)] - mean_x) * (xs[static_cast<size_t>(i)] - mean_x);
      var_y += (ys[static_cast<size_t>(i)] - mean_y) * (ys[static_cast<size_t>(i)] - mean_y);
    }
    const double sx = std::sqrt(var_x / area);
    const double sy = std::sqrt(var_y / area);
    // round enough to be a hole seen from above
    const double roundness = std::min(sx, sy) / std::max({sx, sy, 1e-6});
    if (roundness < 0.45) {
      continue;
    }
    const double score = off + std::abs(area - want_area) / std::max(want_area, 1.0);
    if (score < best_score) {
      best_score = score;
      best = HoleBlob{cu, cv, area};
    }
  }
  return best;
}

// ---------------------------------------------------------------- the known world

// Deleting every point within 5 cm of the arm and everything in the jig volume
// instead left the planner blind exactly where the arm meets a hand: 36 to 55 real
// person voxels within 10 cm of the arm, none of them perceived.
KnownWorld::KnownWorld(mjModel *model, const Calibration &calib) : m_(model), calib_(calib) {
  d_ = mj_makeData(model);
  model->vis.global.fovy = static_cast<float>(calib.fovy_deg);   // the free camera takes its field of view here
  // the offscreen buffer is sized from the model, so it has to hold the picture
  if (model->vis.global.offwidth < calib.width) {
    model->vis.global.offwidth = calib.width;
  }
  if (model->vis.global.offheight < calib.height) {
    model->vis.global.offheight = calib.height;
  }

  // An offscreen OpenGL context, which mjr_makeContext needs before it can render.
  //
  // The context is made on an EGL device, not on EGL_DEFAULT_DISPLAY, and is left
  // surfaceless: MuJoCo renders into its own framebuffer object, and on a machine
  // with two graphics devices the default display renders nothing. This is what
  // MuJoCo's own headless renderer does. MUJOCO_EGL_DEVICE_ID picks the device, as
  // it does there.
  auto query_devices =
      reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
  auto platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
      eglGetProcAddress("eglGetPlatformDisplayEXT"));
  if (query_devices == nullptr || platform_display == nullptr) {
    throw std::runtime_error(
        "this EGL driver has no EGL_EXT_platform_device, which headless rendering needs");
  }
  EGLint device_count = 0;
  query_devices(0, nullptr, &device_count);
  std::vector<EGLDeviceEXT> devices(static_cast<size_t>(std::max(device_count, 0)));
  if (device_count > 0) {
    query_devices(device_count, devices.data(), &device_count);
  }
  int first = 0;
  int last = device_count;
  if (const char *wanted = std::getenv("MUJOCO_EGL_DEVICE_ID")) {
    first = std::atoi(wanted);
    last = first + 1;
    if (first < 0 || first >= device_count) {
      throw std::runtime_error("MUJOCO_EGL_DEVICE_ID is not one of this machine's EGL devices");
    }
  }
  EGLDisplay display = EGL_NO_DISPLAY;
  for (int i = first; i < last; ++i) {
    EGLDisplay candidate = platform_display(EGL_PLATFORM_DEVICE_EXT, devices[static_cast<size_t>(i)],
                                            nullptr);
    if (candidate == EGL_NO_DISPLAY) {
      continue;
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (eglInitialize(candidate, &major, &minor) == EGL_TRUE) {
      display = candidate;
      break;
    }
  }
  if (display == EGL_NO_DISPLAY) {
    throw std::runtime_error("no EGL device could start a display for the known world render");
  }
  const EGLint config_attribs[] = {EGL_RED_SIZE, 8,
                                   EGL_GREEN_SIZE, 8,
                                   EGL_BLUE_SIZE, 8,
                                   EGL_ALPHA_SIZE, 8,
                                   EGL_DEPTH_SIZE, 24,
                                   EGL_STENCIL_SIZE, 8,
                                   EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
                                   EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                   EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                   EGL_NONE};
  EGLConfig config;
  EGLint configs = 0;
  if (eglChooseConfig(display, config_attribs, &config, 1, &configs) != EGL_TRUE || configs < 1) {
    throw std::runtime_error("no EGL configuration for the known world render");
  }
  eglBindAPI(EGL_OPENGL_API);
  EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, nullptr);
  if (context == EGL_NO_CONTEXT) {
    throw std::runtime_error("no EGL context for the known world render");
  }
  if (eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context) != EGL_TRUE) {
    throw std::runtime_error("could not make the EGL context current");
  }
  egl_display_ = display;
  egl_context_ = context;

  mjv_defaultOption(&opt_);
  for (int g = 0; g < 6; ++g) {
    opt_.geomgroup[g] = (g == 0 || g == 1 || g == 2) ? 1 : 0;
  }
  const Vec3 forward = -calib.R.col(2);
  mjv_defaultCamera(&cam_);
  cam_.type = mjCAMERA_FREE;
  cam_.distance = 1.0;
  const Vec3 lookat = calib.pos + forward;
  cam_.lookat[0] = lookat[0];
  cam_.lookat[1] = lookat[1];
  cam_.lookat[2] = lookat[2];
  cam_.azimuth = std::atan2(forward[1], forward[0]) * 180.0 / M_PI;
  cam_.elevation = std::asin(std::min(std::max(forward[2], -1.0), 1.0)) * 180.0 / M_PI;

  mjv_defaultScene(&scn_);
  mjv_makeScene(m_, &scn_, 10000);
  mjr_defaultContext(&con_);
  mjr_makeContext(m_, &con_, mjFONTSCALE_150);
  mjr_setBuffer(mjFB_OFFSCREEN, &con_);
  // the reversed depth map, which is what the coefficients below undo. Left on the
  // standard map every pixel came back at the near plane.
  con_.readDepthMap = mjDEPTH_ZEROFAR;
  open_ = true;
  if (con_.currentBuffer != mjFB_OFFSCREEN) {
    throw std::runtime_error("MuJoCo has no offscreen framebuffer on this GL context");
  }

  for (const char *name : kParts) {
    const int body = mj_name2id(m_, mjOBJ_BODY, name);
    mocap_[name] = body >= 0 ? m_->body_mocapid[body] : -1;
  }
}

KnownWorld::~KnownWorld() { close(); }

void KnownWorld::close() {
  if (!open_) {
    return;
  }
  mjr_freeContext(&con_);
  mjv_freeScene(&scn_);
  if (d_ != nullptr) {
    mj_deleteData(d_);
    d_ = nullptr;
  }
  if (egl_display_ != nullptr) {
    // the display is left running: MuJoCo's own renderer keeps one per process and
    // terminating it would pull the floor from under anything else holding it
    EGLDisplay display = static_cast<EGLDisplay>(egl_display_);
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (egl_context_ != nullptr) {
      eglDestroyContext(display, static_cast<EGLContext>(egl_context_));
    }
    eglReleaseThread();
    egl_display_ = nullptr;
    egl_context_ = nullptr;
  }
  open_ = false;
}

Grid KnownWorld::depth(const Vec6 &q, const std::map<std::string, Vec3> &parts) {
  for (int j = 0; j < kNq; ++j) {
    d_->qpos[j] = q[j];
  }
  for (const auto &[name, slot] : mocap_) {
    if (slot < 0) {
      continue;
    }
    const auto it = parts.find(name);
    const Vec3 where = it == parts.end() ? Vec3(0.0, 0.0, -10.0) : it->second;
    d_->mocap_pos[3 * slot] = where[0];
    d_->mocap_pos[3 * slot + 1] = where[1];
    d_->mocap_pos[3 * slot + 2] = where[2];
    d_->mocap_quat[4 * slot] = 1.0;
    d_->mocap_quat[4 * slot + 1] = 0.0;
    d_->mocap_quat[4 * slot + 2] = 0.0;
    d_->mocap_quat[4 * slot + 3] = 0.0;
  }
  mj_kinematics(m_, d_);
  mjv_updateScene(m_, d_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);

  // Segmented rendering makes the depth accurate at far distances, as MuJoCo's own
  // renderer does it.
  const int was_segment = scn_.flags[mjRND_SEGMENT];
  const int was_idcolor = scn_.flags[mjRND_IDCOLOR];
  scn_.flags[mjRND_SEGMENT] = 1;
  scn_.flags[mjRND_IDCOLOR] = 1;
  mjrRect viewport{0, 0, calib_.width, calib_.height};
  mjr_render(viewport, &scn_, &con_);
  std::vector<float> raw(static_cast<size_t>(calib_.width) * calib_.height);
  mjr_readPixels(nullptr, raw.data(), viewport, &con_);
  scn_.flags[mjRND_SEGMENT] = was_segment;
  scn_.flags[mjRND_IDCOLOR] = was_idcolor;

  // Undo the OpenGL projection. The coefficients are worked out in float, the way
  // glFrustum builds them, and the division in double, which the precision needs.
  // MuJoCo renders in reverse Z, so window coordinates are already normalised.
  const float extent = static_cast<float>(m_->stat.extent);
  const float znear = m_->vis.map.znear * extent;
  const float zfar = m_->vis.map.zfar * extent;
  float c_coef = -(zfar + znear) / (zfar - znear);
  float d_coef = -(2.0f * zfar * znear) / (zfar - znear);
  c_coef = -0.5f * c_coef - 0.5f;
  d_coef = -0.5f * d_coef;

  Grid out(calib_.height, calib_.width);
  for (int v = 0; v < calib_.height; ++v) {
    // EGL hands the picture back bottom row first
    const int src = calib_.height - 1 - v;
    for (int u = 0; u < calib_.width; ++u) {
      const double value = raw[static_cast<size_t>(src) * calib_.width + u];
      const double z = static_cast<double>(d_coef) / (value + static_cast<double>(c_coef));
      out(v, u) = z > 20.0 ? kInf : z;      // nothing of ours on this pixel
    }
  }
  return out;
}

// ---------------------------------------------------------------- the eyes pipeline

Perception::Perception(const Calibration &calib, ToolKinematics *kin, const Taught &taught,
                       double voxel, double unseen_depth, KnownWorld *known)
    : calib_(calib),
      kin_(kin),
      taught_(taught),
      known_(known),
      voxel_(voxel),
      unseen_depth_(unseen_depth) {
  rays_ = calib_.rays();
  const VecX cosines = calib_.axis_cos();
  range_scale_ = cosines.cwiseInverse();
  robot_shapes_ = robot_shapes();
}

std::vector<Perception::RobotShape> Perception::robot_shapes() const {
  const mjModel *m = kin_->model();
  std::vector<RobotShape> out;
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_group[g] != 3) {
      continue;
    }
    out.push_back(RobotShape{g, m->geom_type[g],
                             Vec3(m->geom_size[3 * g], m->geom_size[3 * g + 1],
                                  m->geom_size[3 * g + 2])});
  }
  return out;
}

// How far in front of the empty station a point has to be to count is set per pixel:
// three sigma of the camera's noise at that distance, plus, at a depth edge, how much
// nearer anything within 3 px is. The lateral noise hands a pixel a neighbour's depth,
// so beside the jig plate the worktop reads 16 mm nearer; with two sigma of axial
// noise on top that crossed a fixed 40 mm gate once in a few hundred frames, and two
// such pixels made a person cell that sent the robot away from a corner of the jig.
void Perception::set_background(const Grid &depth) {
  const int h = static_cast<int>(depth.rows());
  const int w = static_cast<int>(depth.cols());
  Grid background(h, w);
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      background(v, u) = depth(v, u) > 0 ? depth(v, u) : kInf;
    }
  }
  const Grid nearest = min_filter(background, kLateralReachPx);
  background_ = VecX(static_cast<Eigen::Index>(h) * w);
  background_gate_ = VecX(static_cast<Eigen::Index>(h) * w);
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      const Eigen::Index i = static_cast<Eigen::Index>(v) * w + u;
      const double z = background(v, u);
      const double lateral = z - nearest(v, u);
      const double gate = std::max(kBackgroundMarginM,
                                   kBackgroundSigmas * depth_sigma(z) +
                                       (std::isfinite(lateral) ? lateral : 0.0));
      background_[i] = z;
      background_gate_[i] = z - gate;
    }
  }
  has_background_ = true;
}

std::map<std::string, Vec3> Perception::part_origins(
    const std::vector<std::string> &present) const {
  Vec3 base = taught_.part_pose;
  if (last_box_.has_value()) {
    base[0] = last_box_->x;
    base[1] = last_box_->y;
  }
  const std::map<std::string, double> dz = {{"wp_base", 0.0},
                                            {"wp_rail", taught_.rail_origin_dz},
                                            {"wp_cover", taught_.cover_origin_dz}};
  std::map<std::string, Vec3> out;
  for (const std::string &name : present) {
    const auto it = dz.find(name);
    if (it != dz.end()) {
      out[name] = base + Vec3(0.0, 0.0, it->second);
    }
  }
  return out;
}

// The camera is a frame late, so subtracting the arm where it is now leaves a band
// along every moving link that reads as a person hugging the arm: the robot froze on
// its own ghost for 18 s. The planner knows its own joint history, so it compares like
// with like and uses q_at_capture.
PersonModel Perception::update(const Grid &depth, const Vec6 &q, double t,
                               const std::vector<std::string> &parts_present,
                               const Vec6 *q_at_capture) {
  const Vec6 q_seen = q_at_capture == nullptr ? q : *q_at_capture;
  const int h = static_cast<int>(depth.rows());
  const int w = static_cast<int>(depth.cols());
  const double *flat = depth.data();
  std::vector<int> idx;
  idx.reserve(static_cast<size_t>(h) * w / 8);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(h) * w; ++i) {
    if (flat[i] <= 0) {
      continue;
    }
    if (has_background_ && !(flat[i] < background_gate_[i])) {
      continue;
    }
    idx.push_back(static_cast<int>(i));
  }
  if (idx.empty()) {
    has_prev_ = false;
    prev_t_ = t;
    PersonModel empty;
    empty.t = t;
    return empty;
  }

  Points pts(static_cast<Eigen::Index>(idx.size()), 3);
  for (size_t k = 0; k < idx.size(); ++k) {
    const int i = idx[k];
    const double range = flat[i] * range_scale_[i];
    pts.row(static_cast<Eigen::Index>(k)) =
        (calib_.pos + rays_.row(i).transpose() * range).transpose();
  }

  std::vector<char> box_mask;
  const std::optional<BoxPose> box_pose = fit_box(pts, &box_mask);
  if (box_pose.has_value()) {
    last_box_ = box_pose;
    last_box_t_ = t;
  }

  std::vector<int> fg_px;
  std::vector<double> fg_z;
  Points kept;
  if (known_ != nullptr) {
    // what the planner knows is there: its own arm and the parts in the jig
    const Grid expected_img = grow_near(known_->depth(q_seen, part_origins(parts_present)),
                                        kSilhouettePx);
    // the arm alone, with the looser tolerance its own edges need
    const Grid arm_img = grow_near(known_->depth(q_seen, {}), kSilhouettePx);
    const double *expected = expected_img.data();
    const double *arm = arm_img.data();
    std::vector<int> keep;
    for (size_t k = 0; k < idx.size(); ++k) {
      const int i = idx[k];
      const double ours = expected[i];
      const double gate = std::max(kBackgroundMarginM, kBackgroundSigmas * depth_sigma(ours));
      bool explained = flat[i] >= ours - gate;
      explained = explained || flat[i] >= arm[i] - kRobotDepthTolM;
      if (!explained) {
        keep.push_back(static_cast<int>(k));
        fg_px.push_back(i);
        fg_z.push_back(flat[i]);
      }
    }
    kept = Points(static_cast<Eigen::Index>(keep.size()), 3);
    for (size_t k = 0; k < keep.size(); ++k) {
      kept.row(static_cast<Eigen::Index>(k)) = pts.row(keep[k]);
    }
    surface_ = VecX(static_cast<Eigen::Index>(h) * w);
    for (Eigen::Index i = 0; i < surface_.size(); ++i) {
      const double empty_cell = has_background_ ? background_[i] : kInf;
      surface_[i] = std::min(empty_cell, expected[i]);
    }
    has_surface_ = true;
  } else {
    // the older filters: kept for comparison, blind next to the arm and over the box
    std::vector<int> keep;
    for (size_t k = 0; k < idx.size(); ++k) {
      if (!box_mask[k]) {
        keep.push_back(static_cast<int>(k));
      }
    }
    Points without_box(static_cast<Eigen::Index>(keep.size()), 3);
    for (size_t k = 0; k < keep.size(); ++k) {
      without_box.row(static_cast<Eigen::Index>(k)) = pts.row(keep[k]);
    }
    if (without_box.rows() > 0) {
      const std::vector<char> mine = is_robot(without_box, q);
      std::vector<int> theirs;
      for (Eigen::Index i = 0; i < without_box.rows(); ++i) {
        if (!mine[static_cast<size_t>(i)]) {
          theirs.push_back(static_cast<int>(i));
        }
      }
      kept = Points(static_cast<Eigen::Index>(theirs.size()), 3);
      for (size_t k = 0; k < theirs.size(); ++k) {
        kept.row(static_cast<Eigen::Index>(k)) = without_box.row(theirs[k]);
      }
    } else {
      kept = without_box;
    }
  }

  PersonModel model;
  model.t = t;
  model.points = static_cast<int>(kept.rows());
  model.box_pose = box_pose;
  model.person_px = fg_px;
  model.person_z = fg_z;
  model.box_pose_held = last_box_;
  model.box_age_s = last_box_t_.has_value() ? t - *last_box_t_ : 0.0;
  if (kept.rows() == 0) {
    has_prev_ = false;
    prev_t_ = t;
    return model;
  }

  std::vector<int64_t> keys;
  Points centres;
  voxelize(kept, &keys, &centres);
  model.voxels = centres;
  track(keys, centres, t, &model.velocity, &model.speed_max);
  model.unseen = unseen_behind(centres);
  prev_keys_ = keys;
  has_prev_ = true;
  prev_t_ = t;
  return model;
}

// The fit looks for the flat face with the most points, not the highest ones: the
// worker's hands are usually on top of the cover, and taking the topmost points put
// the fit 50 mm out and the yaw 80 degrees out. It is coarse on purpose, a few mm, and
// the wrist camera refines it hole by hole.
std::optional<BoxPose> Perception::fit_box(const Points &pts, std::vector<char> *inside) const {
  const Vec3 base = taught_.part_pose;
  inside->assign(static_cast<size_t>(pts.rows()), 0);
  std::vector<int> in;
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    const bool hit = std::abs(pts(i, 0) - base[0]) < 0.13 && std::abs(pts(i, 1) - base[1]) < 0.11 &&
                     pts(i, 2) > base[2] + 0.02 && pts(i, 2) < base[2] + 0.25;
    (*inside)[static_cast<size_t>(i)] = hit ? 1 : 0;
    if (hit) {
      in.push_back(static_cast<int>(i));
    }
  }
  if (in.size() < 80) {
    return std::nullopt;
  }

  const double start = base[2] + 0.02;
  const double stop = base[2] + 0.25;
  const double step = 0.005;
  const int edges = static_cast<int>(std::ceil((stop - start) / step));
  std::vector<int> counts(std::max(edges - 1, 1), 0);
  for (int i : in) {
    const double z = pts(i, 2);
    if (z < start || z > start + (edges - 1) * step) {
      continue;
    }
    int bin = static_cast<int>((z - start) / step);
    bin = std::min(bin, edges - 2);            // the last bin takes its right edge too
    if (bin >= 0) {
      counts[static_cast<size_t>(bin)] += 1;
    }
  }
  const int peak = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
  const double level = start + peak * step + 0.0025;
  // the jig holds the part at a known height, so a face well above it is something
  // lying on the cover, usually a hand
  if (std::abs(level - (base[2] + taught_.cover_top)) > 0.025) {
    return std::nullopt;
  }

  std::vector<int> face;
  for (int i : in) {
    if (std::abs(pts(i, 2) - level) < 0.008) {
      face.push_back(i);
    }
  }
  if (face.size() < 60) {
    return std::nullopt;
  }
  std::vector<double> face_x;
  std::vector<double> face_y;
  std::vector<double> face_z;
  face_x.reserve(face.size());
  face_y.reserve(face.size());
  face_z.reserve(face.size());
  for (int i : face) {
    face_x.push_back(pts(i, 0));
    face_y.push_back(pts(i, 1));
    face_z.push_back(pts(i, 2));
  }
  const double mid_x = median_of(face_x);
  const double mid_y = median_of(face_y);

  const double want_long = std::max(taught_.cover_size_x, taught_.cover_size_y);
  const double want_short = std::min(taught_.cover_size_x, taught_.cover_size_y);
  // Turn the known rectangle until it fits what is on show. Taking the principal
  // axis instead flips by up to 35 degrees as soon as part of the face is hidden.
  // The search covers the jig's own tolerance only: from 3.25 m the cover is about
  // 34 by 26 pixels, which does not pin a rotation, and a free search runs to its
  // own limit. What this fit contributes is the centre and the height.
  double best_score = kInf;
  double best_angle = 0.0;
  Vec3 best_axis = Vec3::Zero();
  Vec3 best_across = Vec3::Zero();
  double best_mid_a = 0.0;
  double best_mid_b = 0.0;
  for (int k = 0; k <= 2 * static_cast<int>(kJigYawDeg); ++k) {
    const double angle = (-kJigYawDeg + k) * M_PI / 180.0;
    const Vec3 axis(std::cos(angle), std::sin(angle), 0.0);
    const Vec3 across(-std::sin(angle), std::cos(angle), 0.0);
    std::vector<double> along_a(face.size());
    std::vector<double> along_b(face.size());
    for (size_t i = 0; i < face.size(); ++i) {
      const double rx = face_x[i] - mid_x;
      const double ry = face_y[i] - mid_y;
      along_a[i] = rx * axis[0] + ry * axis[1];
      along_b[i] = rx * across[0] + ry * across[1];
    }
    const double lo_a = percentile(along_a, 2.0);
    const double hi_a = percentile(along_a, 98.0);
    const double lo_b = percentile(along_b, 2.0);
    const double hi_b = percentile(along_b, 98.0);
    const double score =
        std::abs((hi_a - lo_a) - want_long) + std::abs((hi_b - lo_b) - want_short);
    if (score < best_score) {
      best_score = score;
      best_angle = angle;
      best_axis = axis;
      best_across = across;
      best_mid_a = (lo_a + hi_a) / 2;
      best_mid_b = (lo_b + hi_b) / 2;
    }
  }
  if (best_score > 0.030) {                    // nothing that size is on show
    return std::nullopt;
  }
  const double x = mid_x + best_axis[0] * best_mid_a + best_across[0] * best_mid_b;
  const double y = mid_y + best_axis[1] * best_mid_a + best_across[1] * best_mid_b;
  // The angle the search lands on carries no information at this range: it runs to
  // whichever end of the jig's tolerance fits the noise. The jig sets the yaw.
  return BoxPose{x, y, median_of(face_z), best_angle};
}

std::vector<char> Perception::is_robot(const Points &pts, const Vec6 &q) const {
  const mjModel *m = kin_->model();
  mjData *d = kin_->data();
  for (int j = 0; j < kNq; ++j) {
    d->qpos[j] = q[j];
  }
  mj_kinematics(m, d);
  std::vector<char> mine(static_cast<size_t>(pts.rows()), 0);
  for (const RobotShape &shape : robot_shapes_) {
    const int g = shape.geom;
    const Vec3 p(d->geom_xpos[3 * g], d->geom_xpos[3 * g + 1], d->geom_xpos[3 * g + 2]);
    Mat3 R;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        R(r, c) = d->geom_xmat[9 * g + 3 * r + c];
      }
    }
    for (Eigen::Index i = 0; i < pts.rows(); ++i) {
      if (mine[static_cast<size_t>(i)]) {
        continue;
      }
      const Vec3 local = R.transpose() * (pts.row(i).transpose() - p);
      bool hit = false;
      if (shape.type == mjGEOM_CAPSULE) {
        const double axial = std::min(std::max(local[2], -shape.size[1]), shape.size[1]);
        hit = (local - Vec3(0.0, 0.0, axial)).norm() < shape.size[0] + kRobotMarginM;
      } else if (shape.type == mjGEOM_CYLINDER) {
        const double radial = std::hypot(local[0], local[1]) - shape.size[0];
        const double axial = std::abs(local[2]) - shape.size[1];
        hit = std::hypot(std::max(radial, 0.0), std::max(axial, 0.0)) < kRobotMarginM;
      } else if (shape.type == mjGEOM_BOX) {
        const Vec3 out(std::max(std::abs(local[0]) - shape.size[0], 0.0),
                       std::max(std::abs(local[1]) - shape.size[1], 0.0),
                       std::max(std::abs(local[2]) - shape.size[2], 0.0));
        hit = out.norm() < kRobotMarginM;
      } else {
        hit = local.norm() < shape.size[0] + kRobotMarginM;
      }
      if (hit) {
        mine[static_cast<size_t>(i)] = 1;
      }
    }
  }
  return mine;
}

void Perception::voxelize(const Points &pts, std::vector<int64_t> *keys, Points *centres) const {
  std::vector<int64_t> packed(static_cast<size_t>(pts.rows()));
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    packed[static_cast<size_t>(i)] =
        pack(static_cast<int>(std::floor(pts(i, 0) / voxel_)),
             static_cast<int>(std::floor(pts(i, 1) / voxel_)),
             static_cast<int>(std::floor(pts(i, 2) / voxel_)));
  }
  std::sort(packed.begin(), packed.end());
  keys->clear();
  for (size_t i = 0; i < packed.size();) {
    size_t j = i;
    while (j < packed.size() && packed[j] == packed[i]) {
      ++j;
    }
    if (static_cast<int>(j - i) >= kMinPointsPerVoxel) {
      keys->push_back(packed[i]);
    }
    i = j;
  }
  *centres = Points(static_cast<Eigen::Index>(keys->size()), 3);
  for (size_t i = 0; i < keys->size(); ++i) {
    int x = 0;
    int y = 0;
    int z = 0;
    unpack((*keys)[i], &x, &y, &z);
    (*centres)(static_cast<Eigen::Index>(i), 0) = (x + 0.5) * voxel_;
    (*centres)(static_cast<Eigen::Index>(i), 1) = (y + 0.5) * voxel_;
    (*centres)(static_cast<Eigen::Index>(i), 2) = (z + 0.5) * voxel_;
  }
}

void Perception::track(const std::vector<int64_t> &keys, const Points &centres, double t,
                       Points *velocity, double *speed_max) {
  *velocity = Points::Zero(centres.rows(), 3);
  *speed_max = 0.0;
  if (!has_prev_ || t <= prev_t_ || prev_keys_.empty()) {
    return;
  }
  const double dt = t - prev_t_;
  for (Eigen::Index i = 0; i < centres.rows(); ++i) {
    int x = 0;
    int y = 0;
    int z = 0;
    unpack(keys[static_cast<size_t>(i)], &x, &y, &z);
    double best = kInf;
    Vec3 offset = Vec3::Zero();
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const int64_t probe = pack(x + dx, y + dy, z + dz);
          if (!std::binary_search(prev_keys_.begin(), prev_keys_.end(), probe)) {
            continue;
          }
          // the match sits exactly this far away, so the distance is the offset's own
          const double dist = Vec3(dx, dy, dz).norm() * voxel_;
          if (dist < best) {
            best = dist;
            offset = Vec3(dx, dy, dz) * voxel_;
          }
        }
      }
    }
    if (!std::isfinite(best)) {
      continue;
    }
    const Vec3 vel = offset / dt;
    velocity->row(i) = vel.transpose();
    *speed_max = std::max(*speed_max, vel.norm());
  }
}

Points Perception::unseen_behind(const Points &centres) const {
  if (centres.rows() == 0) {
    return Points(0, 3);
  }
  const int steps = static_cast<int>(unseen_depth_ / voxel_);
  std::vector<int64_t> probe;
  probe.reserve(static_cast<size_t>(centres.rows()) * steps);
  for (Eigen::Index i = 0; i < centres.rows(); ++i) {
    const Vec3 centre = centres.row(i).transpose();
    const Vec3 ray = (centre - calib_.pos).normalized();
    for (int k = 1; k <= steps; ++k) {
      const Vec3 behind = centre + ray * (k * voxel_);
      probe.push_back(pack(static_cast<int>(std::floor(behind[0] / voxel_)),
                           static_cast<int>(std::floor(behind[1] / voxel_)),
                           static_cast<int>(std::floor(behind[2] / voxel_))));
    }
  }
  std::sort(probe.begin(), probe.end());
  probe.erase(std::unique(probe.begin(), probe.end()), probe.end());

  std::vector<int64_t> seen;
  seen.reserve(static_cast<size_t>(centres.rows()));
  for (Eigen::Index i = 0; i < centres.rows(); ++i) {
    seen.push_back(pack(static_cast<int>(std::floor(centres(i, 0) / voxel_)),
                        static_cast<int>(std::floor(centres(i, 1) / voxel_)),
                        static_cast<int>(std::floor(centres(i, 2) / voxel_))));
  }
  std::sort(seen.begin(), seen.end());

  std::vector<Vec3> out;
  out.reserve(probe.size());
  const double f = calib_.focal();
  for (int64_t key : probe) {
    if (std::binary_search(seen.begin(), seen.end(), key)) {
      continue;
    }
    int x = 0;
    int y = 0;
    int z = 0;
    unpack(key, &x, &y, &z);
    const Vec3 point((x + 0.5) * voxel_, (y + 0.5) * voxel_, (z + 0.5) * voxel_);
    if (has_surface_) {
      // A hand can hide behind the person, but not inside the bench or behind the
      // box: keep only what lies in front of the nearest known surface on its ray.
      const Vec3 rel = calib_.R.transpose() * (point - calib_.pos);
      const double axis_depth = -rel[2];
      if (axis_depth <= 1e-3) {
        continue;
      }
      const int u = static_cast<int>(
          std::nearbyint(calib_.width / 2.0 - 0.5 + f * rel[0] / axis_depth));
      const int v = static_cast<int>(
          std::nearbyint(calib_.height / 2.0 - 0.5 - f * rel[1] / axis_depth));
      if (u < 0 || u >= calib_.width || v < 0 || v >= calib_.height) {
        continue;
      }
      if (!(axis_depth < surface_[static_cast<Eigen::Index>(v) * calib_.width + u] - 0.5 * voxel_)) {
        continue;
      }
    }
    out.push_back(point);
  }
  Points result(static_cast<Eigen::Index>(out.size()), 3);
  for (size_t i = 0; i < out.size(); ++i) {
    result.row(static_cast<Eigen::Index>(i)) = out[i].transpose();
  }
  return result;
}

}  // namespace sightline
