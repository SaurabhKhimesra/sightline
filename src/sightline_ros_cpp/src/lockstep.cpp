#include "sightline_ros/lockstep.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <sensor_msgs/msg/point_field.hpp>

#include "sightline_ros/caption_font.hpp"

namespace sightline_ros {
namespace {

using sightline::Grid;
using sightline::Points;
using sightline::Vec3;

// Float mask of text in the 5 x 7 font, px pixels per font pixel, anti-aliased the
// same way the sim draws it: doubled, box blurred over 3 pixels, then halved again.
std::vector<double> text_mask(const std::string &text, int px, int *out_h, int *out_w) {
  const auto &font = caption_font();
  const auto space = font.find(' ');
  std::vector<std::vector<int>> cells;          // 7 rows of the whole line
  cells.assign(7, {});
  for (char raw : text) {
    const char ch = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
    auto glyph = font.find(ch);
    if (glyph == font.end()) {
      glyph = space;
    }
    for (int r = 0; r < 7; ++r) {
      const char *bits = glyph->second[static_cast<size_t>(r)];
      for (int c = 0; c < 5; ++c) {
        cells[static_cast<size_t>(r)].push_back(bits[c] == '1' ? 1 : 0);
      }
      cells[static_cast<size_t>(r)].push_back(0);   // one blank column between glyphs
    }
  }
  const int cols = cells[0].empty() ? 1 : static_cast<int>(cells[0].size());
  const int rows = 7 + 3;                        // three blank rows under the line
  // each font pixel becomes px * 2 screen pixels, blurred, then sampled every other one
  const int big_h = rows * px * 2;
  const int big_w = cols * px * 2;
  std::vector<double> big(static_cast<size_t>(big_h) * big_w, 0.0);
  for (int r = 0; r < 7; ++r) {
    for (int c = 0; c < cols && c < static_cast<int>(cells[static_cast<size_t>(r)].size()); ++c) {
      if (!cells[static_cast<size_t>(r)][static_cast<size_t>(c)]) {
        continue;
      }
      for (int dv = 0; dv < px * 2; ++dv) {
        for (int du = 0; du < px * 2; ++du) {
          big[static_cast<size_t>(r * px * 2 + dv) * big_w + (c * px * 2 + du)] = 1.0;
        }
      }
    }
  }
  // a 3 by 3 box blur, edges held
  std::vector<double> blurred(big.size(), 0.0);
  for (int v = 0; v < big_h; ++v) {
    for (int u = 0; u < big_w; ++u) {
      double sum = 0.0;
      for (int dv = -1; dv <= 1; ++dv) {
        for (int du = -1; du <= 1; ++du) {
          const int vv = std::min(std::max(v + dv, 0), big_h - 1);
          const int uu = std::min(std::max(u + du, 0), big_w - 1);
          sum += big[static_cast<size_t>(vv) * big_w + uu];
        }
      }
      blurred[static_cast<size_t>(v) * big_w + u] = sum / 9.0;
    }
  }
  *out_h = (big_h + 1) / 2;
  *out_w = (big_w + 1) / 2;
  std::vector<double> out(static_cast<size_t>(*out_h) * *out_w, 0.0);
  for (int v = 0; v < *out_h; ++v) {
    for (int u = 0; u < *out_w; ++u) {
      out[static_cast<size_t>(v) * *out_w + u] = blurred[static_cast<size_t>(2 * v) * big_w + 2 * u];
    }
  }
  return out;
}

}  // namespace

const std::map<std::string, std::string> &topics() {
  static const std::map<std::string, std::string> all = {
      // what the cell knows once, sent once and kept for a late subscriber
      {"calibration", "/sightline/cell/calibration"},  // where the planner may believe the eyes camera is
      {"background", "/eyes/background"},              // the empty station's depth, taken at commissioning
      {"blind", "/sightline/cell/blind_cells"},        // cells the furniture hides from the camera
      {"cell", "/sightline/cell/taught"},              // the jig, the holes, the feeder, the park pose
      // every 10 ms
      {"joints", "/joint_states"},
      {"tick", "/sightline/tick"},
      {"cmd", "/sightline/cmd"},
      {"clock", "/clock"},
      // every 40 ms
      {"depth", "/eyes/depth"},
      {"frame", "/sightline/frame"},
      {"image", "/eyes/image"},
      {"judge", "/sightline/judge"},
      {"worker", "/sightline/worker"},
      {"station", "/sightline/station"},
      {"caption", "/sightline/caption_image"},
      // the planner's view of things, for rviz
      {"grid", "/sightline/grid"},
      {"arm", "/sightline/arm_points"},
      {"target", "/sightline/target"},
      {"planner_caption", "/sightline/planner_caption"},
      {"ready", "/sightline/planner/ready"},   // the planner has built its world and answers ticks
  };
  return all;
}

const std::string &topic(const std::string &name) { return topics().at(name); }

rclcpp::QoS latched() {
  rclcpp::QoS qos(1);
  qos.transient_local();
  qos.reliable();
  return qos;
}

builtin_interfaces::msg::Time stamp(double t) {
  builtin_interfaces::msg::Time out;
  out.sec = static_cast<int32_t>(t);
  out.nanosec = static_cast<uint32_t>(std::llround((t - out.sec) * 1e9));
  return out;
}

std_msgs::msg::String to_json(const Json &value) {
  std_msgs::msg::String msg;
  msg.data = value.dump();
  return msg;
}

Json from_json(const std_msgs::msg::String &msg) { return Json::parse(msg.data); }

sensor_msgs::msg::Image depth_to_msg(const Grid &depth, double t, const std::string &frame_id) {
  sensor_msgs::msg::Image msg;
  msg.header.stamp = stamp(t);
  msg.header.frame_id = frame_id;
  msg.height = static_cast<uint32_t>(depth.rows());
  msg.width = static_cast<uint32_t>(depth.cols());
  msg.encoding = "32FC1";
  msg.is_bigendian = 0;
  msg.step = msg.width * 4;
  msg.data.resize(static_cast<size_t>(msg.height) * msg.step);
  auto *out = reinterpret_cast<float *>(msg.data.data());
  for (Eigen::Index i = 0; i < depth.size(); ++i) {
    out[i] = static_cast<float>(depth.data()[i]);
  }
  return msg;
}

Grid msg_to_depth(const sensor_msgs::msg::Image &msg) {
  if (msg.encoding != "32FC1") {
    throw std::runtime_error("the depth picture is " + msg.encoding + ", not 32FC1");
  }
  Grid out(msg.height, msg.width);
  const auto *in = reinterpret_cast<const float *>(msg.data.data());
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out.data()[i] = in[i];
  }
  return out;
}

sensor_msgs::msg::Image caption_image(
    const std::vector<std::pair<std::string, Vec3>> &lines, double t, int width, int px) {
  const int height = 8 + 12 * px * static_cast<int>(lines.size());
  std::vector<float> strip(static_cast<size_t>(height) * width * 3, 0.10f);
  for (size_t n = 0; n < lines.size(); ++n) {
    int mh = 0;
    int mw = 0;
    const std::vector<double> mask = text_mask(lines[n].first, px, &mh, &mw);
    const int x = 12;
    const int y = 8 + static_cast<int>(n) * 12 * px;
    for (int v = 0; v < mh && y + v < height; ++v) {
      for (int u = 0; u < mw && x + u < width; ++u) {
        const double a = mask[static_cast<size_t>(v) * mw + u];
        if (a <= 0.0) {
          continue;
        }
        const size_t at = (static_cast<size_t>(y + v) * width + (x + u)) * 3;
        for (int c = 0; c < 3; ++c) {
          strip[at + c] = static_cast<float>(strip[at + c] * (1 - a) + lines[n].second[c] * a);
        }
      }
    }
  }
  sensor_msgs::msg::Image msg;
  msg.header.stamp = stamp(t);
  msg.header.frame_id = "caption";
  msg.height = static_cast<uint32_t>(height);
  msg.width = static_cast<uint32_t>(width);
  msg.encoding = "rgb8";
  msg.is_bigendian = 0;
  msg.step = msg.width * 3;
  msg.data.resize(strip.size());
  for (size_t i = 0; i < strip.size(); ++i) {
    msg.data[i] = static_cast<uint8_t>(
        std::min(std::max(std::lround(strip[i] * 255.0), 0L), 255L));
  }
  return msg;
}

sensor_msgs::msg::PointCloud2 cloud_msg(const Points &points, const Vec3 &rgb, double t,
                                        const std::string &frame_id) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = stamp(t);
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(points.rows());
  const char *names[4] = {"x", "y", "z", "rgb"};
  for (int i = 0; i < 4; ++i) {
    sensor_msgs::msg::PointField field;
    field.name = names[i];
    field.offset = static_cast<uint32_t>(4 * i);
    field.datatype = i == 3 ? sensor_msgs::msg::PointField::UINT32
                            : sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    msg.fields.push_back(field);
  }
  msg.is_bigendian = false;
  msg.point_step = 16;
  msg.row_step = 16 * msg.width;
  msg.is_dense = true;
  msg.data.resize(static_cast<size_t>(msg.row_step));
  const uint32_t colour =
      (static_cast<uint32_t>(std::lround(rgb[0] * 255)) << 16) |
      (static_cast<uint32_t>(std::lround(rgb[1] * 255)) << 8) |
      static_cast<uint32_t>(std::lround(rgb[2] * 255));
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    const float xyz[3] = {static_cast<float>(points(i, 0)), static_cast<float>(points(i, 1)),
                          static_cast<float>(points(i, 2))};
    uint8_t *at = msg.data.data() + static_cast<size_t>(i) * 16;
    std::memcpy(at, xyz, 12);
    std::memcpy(at + 12, &colour, 4);
  }
  return msg;
}

}  // namespace sightline_ros
