// The planner as a ROS 2 node: B4 on the camera's grid, in lockstep with the cell
// node on sim time.
//
//     ros2 run sightline_ros_cpp planner_node --ros-args
//         -p robot_model:=<robot_only.xml> -p seed:=0 -p out:=results/ros2/stage1
//
// It waits for what the cell publishes once (the calibration it may believe, the empty
// station's depth, the blind cells, the taught points), then answers every
// /sightline/tick with a /sightline/cmd. A tick naming a picture it has not processed
// yet waits for that picture. For rviz it publishes the grid's off-limits cells, the
// 63 points of the arm it checks, the hole it is going for and a caption strip.
//
// It reads nothing from the simulation: what it knows of the cell arrives on topics,
// and its own arm comes from the MJCF that station.export_robot_mjcf writes.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "sightline_planner/b4.hpp"
#include "sightline_planner/motion.hpp"
#include "sightline_planner/perceive.hpp"
#include "sightline_planner/settings.hpp"
#include "sightline_planner/viewgrid.hpp"
#include "sightline_ros/lockstep.hpp"

namespace sightline_ros {
namespace {

using sightline::Grid;
using sightline::Points;
using sightline::Vec3;
using sightline::Vec6;
using visualization_msgs::msg::Marker;

struct ModelDeleter {
  void operator()(mjModel *m) const {
    if (m != nullptr) {
      mj_deleteModel(m);
    }
  }
};
using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;

ModelPtr load_model(const std::string &path) {
  char error[1000] = "";
  mjModel *m = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
  if (m == nullptr) {
    throw std::runtime_error("could not load the robot model at " + path + ": " + error);
  }
  return ModelPtr(m);
}

double median_of(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

double percentile_of(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double pos = p / 100.0 * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  return lo == hi ? values[lo] : values[lo] + (pos - lo) * (values[hi] - values[lo]);
}

std::vector<double> tail(const std::vector<double> &values, size_t n) {
  if (values.size() <= n) {
    return values;
  }
  return std::vector<double>(values.end() - static_cast<long>(n), values.end());
}

double round_to(double value, int places) {
  const double scale = std::pow(10.0, places);
  return std::round(value * scale) / scale;
}

double ms_since(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string today() {
  const std::time_t now = std::time(nullptr);
  std::tm parts{};
  localtime_r(&now, &parts);
  char buffer[16];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &parts);
  return buffer;
}

// One log line as a YAML sequence, the shape the Python runner writes.
template <typename... Cells>
YAML::Node row_of(const Cells &...cells) {
  YAML::Node row(YAML::NodeType::Sequence);
  (row.push_back(cells), ...);
  return row;
}

YAML::Node summary_to_yaml(const sightline::Summary &s) {
  YAML::Node out;
  out["screws_driven"] = s.screws_driven;
  out["screws_total"] = s.screws_total;
  for (const auto &[name, deg] : s.turn_angles_deg) {
    out["turn_angles_deg"][name] = deg;
  }
  out["hole_seen_frames"] = s.hole_seen_frames;
  out["hole_missed_frames"] = s.hole_missed_frames;
  out["unreachable"] = s.unreachable;
  out["hole_rejected_corrections"] = s.hole_rejected_corrections;
  for (const auto &y : s.yields) {
    out["yields"].push_back(row_of(y.from, y.to, y.screw));
  }
  for (const auto &m : s.missed) {
    out["missed"].push_back(row_of(m.t, m.screw, m.why));
  }
  out["hole_error_mm"] = s.hole_error_mm;
  for (const auto &e : s.state_log) {
    out["state_log"].push_back(row_of(e.t, e.state, e.screw));
  }
  for (const auto &c : s.choices) {
    out["choices"].push_back(row_of(c.t, c.screw, c.turn_deg, c.why));
  }
  for (const auto &[name, n] : s.options_per_screw) {
    out["options_per_screw"][name] = n;
  }
  for (const auto &[name, n] : s.options_view_clear) {
    out["options_view_clear"][name] = n;
  }
  for (const auto &g : s.gave_up) {
    out["gave_up"].push_back(row_of(g.t, g.screw, g.state));
  }
  for (const auto &w : s.swaps) {
    out["swaps"].push_back(row_of(w.t, w.screw, w.why));
  }
  if (s.has_guard) {
    for (const auto &[name, n] : s.guard_verdicts) {
      out["guard"]["verdicts"][name] = n;
    }
    out["guard"]["ms_median"] = s.guard_ms_median;
    out["guard"]["ms_p99"] = s.guard_ms_p99;
    out["guard"]["ms_max"] = s.guard_ms_max;
    out["guard"]["points_on_the_arm"] = s.points_on_the_arm;
  }
  out["feeder_turn_deg"] = s.feeder_turn_deg;
  if (s.has_feeder_clearance) {
    out["feeder_arm_behind_jig_m"] = s.feeder_arm_behind_jig_m;
  }
  return out;
}

class Planner : public rclcpp::Node {
 public:
  Planner() : rclcpp::Node("sightline_planner") {
    seed_ = declare_parameter<int>("seed", 0);
    out_dir_ = declare_parameter<std::string>("out", "results/ros2/stage1");
    model_path_ = declare_parameter<std::string>("robot_model", "");
    if (model_path_.empty()) {
      throw std::runtime_error(
          "set robot_model to the MJCF the planner plans with "
          "(sightline_sim.station.export_robot_mjcf writes it)");
    }

    using std_msgs::msg::String;
    using sensor_msgs::msg::Image;
    sub_calibration_ = create_subscription<String>(
        topic("calibration"), latched(),
        [this](String::SharedPtr m) { learn("calibration", from_json(*m)); });
    sub_background_ = create_subscription<Image>(
        topic("background"), latched(),
        [this](Image::SharedPtr m) { background_ = msg_to_depth(*m); learn("background", {}); });
    sub_blind_ = create_subscription<String>(
        topic("blind"), latched(), [this](String::SharedPtr m) { learn("blind", from_json(*m)); });
    sub_cell_ = create_subscription<String>(
        topic("cell"), latched(), [this](String::SharedPtr m) { learn("cell", from_json(*m)); });
    sub_depth_ = create_subscription<Image>(
        topic("depth"), 5, [this](Image::SharedPtr m) { on_depth(*m); });
    sub_frame_ = create_subscription<String>(
        topic("frame"), 10, [this](String::SharedPtr m) { on_frame(*m); });
    sub_tick_ = create_subscription<String>(
        topic("tick"), 50, [this](String::SharedPtr m) { on_tick(*m); });

    pub_cmd_ = create_publisher<String>(topic("cmd"), 50);
    for (const char *name : {"seen", "remembered", "behind"}) {
      pub_grid_[name] =
          create_publisher<sensor_msgs::msg::PointCloud2>(topic("grid") + "/" + name, 5);
    }
    pub_ready_ = create_publisher<String>(topic("ready"), latched());
    pub_arm_ = create_publisher<Marker>(topic("arm"), 5);
    pub_target_ = create_publisher<Marker>(topic("target"), 5);
    pub_caption_ = create_publisher<Image>(topic("planner_caption"), 5);
  }

  bool done() const { return done_; }

 private:
  // ------------------------------------------------------------ what the cell knows
  void learn(const std::string &key, const Json &value) {
    if (knowledge_.count(key)) {
      return;
    }
    knowledge_[key] = value;
    RCLCPP_INFO(get_logger(), "received the cell's %s", key.c_str());
    const bool all = knowledge_.count("calibration") && knowledge_.count("background") &&
                     knowledge_.count("blind") && knowledge_.count("cell");
    if (!ready_ && all) {
      setup();
    }
  }

  void setup() {
    const Json &cal = knowledge_["calibration"];
    const Json &cell = knowledge_["cell"];
    fps_ = cell["fps"].get<double>();
    calib_.pos = Vec3(cal["pos"][0], cal["pos"][1], cal["pos"][2]);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        calib_.R(r, c) = cal["R"][r][c].get<double>();
      }
    }
    calib_.width = cal["width"].get<int>();
    calib_.height = cal["height"].get<int>();
    calib_.fovy_deg = cal["fovy_deg"].get<double>();

    arm_model_ = load_model(model_path_);
    render_model_ = load_model(model_path_);
    kin_ = std::make_unique<sightline::ToolKinematics>(arm_model_.get());

    taught_.part_pose = Vec3(cell["part_pose"][0], cell["part_pose"][1], cell["part_pose"][2]);
    for (const auto &h : cell["rail_holes"]) {
      taught_.rail_holes.emplace_back(h[0], h[1], h[2]);
    }
    for (const auto &h : cell["cover_holes"]) {
      taught_.cover_holes.emplace_back(h[0], h[1], h[2]);
    }
    taught_.feeder_pick = Vec3(cell["feeder_pick"][0], cell["feeder_pick"][1],
                               cell["feeder_pick"][2]);
    const double zw = cell["worktop_z"].get<double>();
    Vec6 park;
    for (int j = 0; j < 6; ++j) {
      park[j] = cell["park"][j].get<double>();
    }
    planes_.push_back(sightline::Plane{Vec3(0.0, 0.0, zw), Vec3(0.0, 0.0, 1.0)});

    known_ = std::make_unique<sightline::KnownWorld>(render_model_.get(), calib_);
    grid_ = std::make_unique<sightline::ViewGrid>(calib_, sightline::kCellPx, sightline::kSafeM,
                                                  sightline::kBehindM, "measured");
    const Vec3 park_tip(-0.45, 0.33, zw + 0.30);
    b4_ = std::make_unique<sightline::B4>(kin_.get(), taught_, park_tip, grid_.get(), planes_);
    const Vec3 eye = calib_.pos;
    b4_->build(&park, &eye);
    motion_ = std::make_unique<sightline::MotionLayer>(kin_.get(), false, false);
    b4_->set_motion(motion_.get());
    per_ = std::make_unique<sightline::Perception>(calib_, kin_.get(), taught_,
                                                   sightline::kVoxelM,
                                                   sightline::kUnseenDepthM, known_.get());
    per_->set_background(background_);
    grid_->set_background(background_);

    const auto &cells = knowledge_["blind"]["cells"];
    hidden_ = Points(static_cast<Eigen::Index>(cells.size()), 3);
    for (size_t i = 0; i < cells.size(); ++i) {
      hidden_(static_cast<Eigen::Index>(i), 0) = cells[i][0].get<double>();
      hidden_(static_cast<Eigen::Index>(i), 1) = cells[i][1].get<double>();
      hidden_(static_cast<Eigen::Index>(i), 2) = cells[i][2].get<double>();
    }
    ready_ = true;
    RCLCPP_INFO(get_logger(), "planner ready: %ld blind cells, worktop at %.3f m",
                static_cast<long>(hidden_.rows()), zw);
    Json note;
    note["ready"] = true;
    note["blind_cells"] = static_cast<int>(hidden_.rows());
    pub_ready_->publish(to_json(note));
    if (pending_.has_value()) {
      const Json tick = *pending_;
      pending_.reset();
      on_tick_ready(tick);
    }
  }

  // A blind cell counts only if a seen part of him is close enough to have put a
  // hand in it (the gate 5 runner's rule).
  Points live_blind() const {
    const Points person = grid_->seen_points();
    if (person.rows() == 0 || hidden_.rows() == 0) {
      return Points(0, 3);
    }
    std::vector<Eigen::Index> near;
    for (Eigen::Index i = 0; i < hidden_.rows(); ++i) {
      double gap = std::numeric_limits<double>::infinity();
      for (Eigen::Index j = 0; j < person.rows(); ++j) {
        gap = std::min(gap, (hidden_.row(i) - person.row(j)).norm());
      }
      if (gap < sightline::kHandReachM) {
        near.push_back(i);
      }
    }
    Points out(static_cast<Eigen::Index>(near.size()), 3);
    for (size_t i = 0; i < near.size(); ++i) {
      out.row(static_cast<Eigen::Index>(i)) = hidden_.row(near[i]);
    }
    return out;
  }

  // ------------------------------------------------------------ pictures
  static int64_t key_of(double t) { return std::llround(t * 1000); }

  void on_depth(const sensor_msgs::msg::Image &msg) {
    const double t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
    depths_[key_of(t)] = msg_to_depth(msg);
    try_frame(key_of(t));
  }

  void on_frame(const std_msgs::msg::String &msg) {
    const Json info = from_json(msg);
    frames_[key_of(info["t"].get<double>())] = info;
    try_frame(key_of(info["t"].get<double>()));
  }

  void try_frame(int64_t key) {
    if (!ready_ || !depths_.count(key) || !frames_.count(key)) {
      return;
    }
    const Json info = frames_[key];
    const Grid depth = depths_[key];
    frames_.erase(key);
    depths_.erase(key);
    process_frame(info, depth);
  }

  void process_frame(const Json &info, const Grid &depth) {
    const auto t0 = std::chrono::steady_clock::now();
    const double t = info["t"].get<double>();
    const int index = info["index"].get<int>();
    Vec6 q;
    Vec6 seen_q;
    for (int j = 0; j < 6; ++j) {
      q[j] = info["q"][j].get<double>();
      seen_q[j] = info["seen_q"][j].get<double>();
    }
    const sightline::PersonModel model =
        per_->update(depth, q, t,
                     b4_->parts_in_jig(info["signal"].get<std::string>(),
                                       per_->last_box().has_value()),
                     &seen_q);
    const auto t1 = std::chrono::steady_clock::now();
    b4_->see(model.person_px, model.person_z, info["taken_at"].get<double>(), seen_q);
    const auto t2 = std::chrono::steady_clock::now();
    b4_->guard()->set_blind(live_blind());
    const auto t3 = std::chrono::steady_clock::now();
    last_frame_ = index;
    frames_seen_ += 1;
    frame_ms_.push_back(ms_since(t0));
    perceive_ms_.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    grid_ms_.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    blind_ms_.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    publish_grid(t);
    rviz_ms_.push_back(ms_since(t3));
    if (pending_.has_value() && (*pending_)["frame"].get<int>() <= index) {
      const Json tick = *pending_;
      pending_.reset();
      on_tick_ready(tick);
    }
  }

  // ------------------------------------------------------------ ticks
  void on_tick(const std_msgs::msg::String &msg) {
    const Json tick = from_json(msg);
    if (tick.contains("end") && tick["end"].get<bool>()) {
      finish();
      return;
    }
    if (!ready_ || tick["frame"].get<int>() > last_frame_) {
      pending_ = tick;               // the picture it names has not arrived yet
      return;
    }
    on_tick_ready(tick);
  }

  void on_tick_ready(const Json &tick) {
    sightline::SensorFrame frame;
    frame.t = tick["t"].get<double>();
    for (int j = 0; j < 6; ++j) {
      frame.q[j] = tick["q"][j].get<double>();
      frame.qd[j] = tick["qd"][j].get<double>();
    }
    frame.jig_signal = tick["signal"].get<std::string>();

    const Vec6 wanted = b4_->task(frame);
    sightline::Obstacles obstacles;
    obstacles.planes = planes_;
    const sightline::Report rep = motion_->solve_joint(frame.q, wanted, obstacles);
    const auto t0 = std::chrono::steady_clock::now();
    const Vec6 qd = b4_->safe_command(frame, rep.qd);
    guard_ms_.push_back(ms_since(t0));
    motion_->set_last(qd);           // the next ramp starts from what was sent
    const std::string verdict = b4_->guard()->last();
    verdicts_[verdict] += 1;
    solve_ms_.push_back(rep.solve_ms);
    ticks_ += 1;

    const std::string screw = b4_->index() < static_cast<int>(b4_->screws().size())
                                  ? b4_->screws()[static_cast<size_t>(b4_->index())].name
                                  : std::string();
    int driven = 0;
    for (const sightline::Screw &s : b4_->screws()) {
      driven += s.driven ? 1 : 0;
    }
    Json out;
    out["tick"] = tick["tick"].get<int>();
    out["qd"] = std::vector<double>(qd.data(), qd.data() + 6);
    out["state"] = b4_->state();
    out["verdict"] = verdict;
    out["want_speed"] = wanted.norm();
    out["screw"] = screw;
    out["screws_done"] = driven;
    out["yield_since"] = b4_->yield_since();
    out["solve_ms"] = rep.solve_ms;
    pub_cmd_->publish(to_json(out));
    if (tick["tick"].get<int>() % 5 == 0) {
      publish_arm(frame.q, frame.t, screw);
    }
  }

  // ------------------------------------------------------------ for rviz
  // The grid's cells as three point clouds: where the camera sees him (red), the
  // cells its own arm hides and it remembers (orange), and the far end of every
  // cell, 15 cm behind him (dark red): the arm keeps off the whole stretch.
  void publish_grid(double t) {
    Points near(0, 3);
    Points remembered(0, 3);
    Points behind(0, 3);
    if (grid_->size() > 0) {
      const Points near_local = grid_->points_at(grid_->rows(), grid_->cols(), grid_->near());
      const Points far_local = grid_->points_at(grid_->rows(), grid_->cols(), grid_->far());
      std::vector<Eigen::Index> seen;
      std::vector<Eigen::Index> hidden;
      for (int i = 0; i < grid_->size(); ++i) {
        (grid_->hidden()[static_cast<size_t>(i)] ? hidden : seen).push_back(i);
      }
      auto to_world = [this](const Points &local, const std::vector<Eigen::Index> &which) {
        Points out(static_cast<Eigen::Index>(which.size()), 3);
        for (size_t i = 0; i < which.size(); ++i) {
          out.row(static_cast<Eigen::Index>(i)) =
              (calib_.R * local.row(which[i]).transpose() + calib_.pos).transpose();
        }
        return out;
      };
      std::vector<Eigen::Index> all(static_cast<size_t>(grid_->size()));
      for (size_t i = 0; i < all.size(); ++i) {
        all[i] = static_cast<Eigen::Index>(i);
      }
      near = to_world(near_local, seen);
      remembered = to_world(near_local, hidden);
      behind = to_world(far_local, all);
    }
    pub_grid_["seen"]->publish(cloud_msg(near, Vec3(0.92, 0.16, 0.12), t));
    pub_grid_["remembered"]->publish(cloud_msg(remembered, Vec3(1.0, 0.59, 0.0), t));
    pub_grid_["behind"]->publish(cloud_msg(behind, Vec3(0.45, 0.05, 0.05), t));

    int remembered_cells = 0;
    for (int i = 0; i < grid_->size(); ++i) {
      remembered_cells += grid_->hidden()[static_cast<size_t>(i)] ? 1 : 0;
    }
    std::string verdict = b4_->guard()->last();
    std::transform(verdict.begin(), verdict.end(), verdict.begin(), ::toupper);
    std::string state = b4_->state();
    std::transform(state.begin(), state.end(), state.begin(), ::toupper);
    char line1[160];
    char line2[240];
    char line3[200];
    std::snprintf(line1, sizeof(line1), "PLANNER B4 ON ROS 2   PICTURE %d   T %5.1f S",
                  frames_seen_, t);
    std::snprintf(line2, sizeof(line2),
                  "STATE %s   GUARD %s   OFF LIMITS CELLS %d (%d REMEMBERED)   "
                  "BLIND CELLS LIVE %ld",
                  state.c_str(), verdict.empty() ? "GO" : verdict.c_str(), grid_->size(),
                  remembered_cells, static_cast<long>(b4_->guard()->blind().rows()));
    std::snprintf(line3, sizeof(line3), "PICTURE %.0f MS   GUARD %.1f MS   QP %.1f MS",
                  median_of(tail(frame_ms_, 25)), median_of(tail(guard_ms_, 100)),
                  median_of(tail(solve_ms_, 100)));
    const std::vector<std::pair<std::string, Vec3>> lines = {
        {line1, Vec3(0.95, 0.95, 0.92)},
        {line2, Vec3(0.55, 0.85, 1.0)},
        {line3, Vec3(0.85, 0.95, 0.85)}};
    pub_caption_->publish(caption_image(lines, t));
  }

  void publish_arm(const Vec6 &q, double t, const std::string &screw) {
    Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = stamp(t);
    mk.ns = "arm";
    mk.id = 0;
    mk.type = Marker::SPHERE_LIST;
    mk.action = Marker::ADD;
    mk.pose.orientation.w = 1.0;
    mk.scale.x = mk.scale.y = mk.scale.z = 0.03;
    mk.color.r = 0.2f;
    mk.color.g = 0.95f;
    mk.color.b = 0.4f;
    mk.color.a = 0.9f;
    const Points pts = b4_->guard()->points(q);
    for (Eigen::Index i = 0; i < pts.rows(); ++i) {
      geometry_msgs::msg::Point p;
      p.x = pts(i, 0);
      p.y = pts(i, 1);
      p.z = pts(i, 2);
      mk.points.push_back(p);
    }
    pub_arm_->publish(mk);

    Marker tg;
    tg.header.frame_id = "world";
    tg.header.stamp = stamp(t);
    tg.ns = "target";
    tg.id = 0;
    tg.type = Marker::SPHERE;
    const bool going = !screw.empty() && (b4_->state() == "to_hole" ||
                                          sightline::at_the_hole(b4_->state()));
    tg.action = going ? Marker::ADD : Marker::DELETE;
    const sightline::Screw *hole = nullptr;
    for (const sightline::Screw &s : b4_->screws()) {
      if (s.name == screw) {
        hole = &s;
      }
    }
    if (going && hole != nullptr) {
      tg.pose.position.x = hole->hole[0];
      tg.pose.position.y = hole->hole[1];
      tg.pose.position.z = hole->hole[2];
    }
    tg.pose.orientation.w = 1.0;
    tg.scale.x = tg.scale.y = tg.scale.z = 0.05;
    tg.color.r = 0.3f;
    tg.color.g = 0.6f;
    tg.color.b = 1.0f;
    tg.color.a = 0.9f;
    pub_target_->publish(tg);
  }

  // ------------------------------------------------------------ the end
  void finish() {
    if (done_) {
      return;
    }
    done_ = true;
    YAML::Node result;
    result["stage"] = 1;
    result["variant"] = "B4";
    result["seed"] = seed_;
    result["date"] = today();
    result["ticks_answered"] = ticks_;
    result["pictures"] = frames_seen_;
    if (!frame_ms_.empty()) {
      result["picture_ms_median"] = round_to(median_of(frame_ms_), 1);
      result["picture_ms_max"] = round_to(*std::max_element(frame_ms_.begin(), frame_ms_.end()), 1);
    }
    const std::map<std::string, const std::vector<double> *> parts = {
        {"perceive", &perceive_ms_}, {"grid", &grid_ms_}, {"blind", &blind_ms_},
        {"rviz", &rviz_ms_}};
    for (const auto &[name, values] : parts) {
      if (!values->empty()) {
        result["picture_parts_ms_median"][name] = round_to(median_of(*values), 1);
      }
    }
    if (!guard_ms_.empty()) {
      result["guard_ms_median"] = round_to(median_of(guard_ms_), 2);
      result["guard_ms_p99"] = round_to(percentile_of(guard_ms_, 99.0), 2);
    }
    if (!solve_ms_.empty()) {
      result["qp_ms_median"] = round_to(median_of(solve_ms_), 2);
    }
    for (const auto &[name, n] : verdicts_) {
      result["guard_verdicts"][name] = n;
    }
    const sightline::Summary summary =
        ready_ ? b4_->summary() : sightline::Summary{};
    result["planner_summary"] = summary_to_yaml(summary);

    std::filesystem::create_directories(out_dir_);
    const std::string path = out_dir_ + "/planner_B4_seed" + std::to_string(seed_) + ".yaml";
    std::ofstream file(path);
    file << result;
    file.close();
    RCLCPP_INFO(get_logger(),
                "PLANNER: %d ticks, %d pictures; picture %.1f ms median / %.1f max, "
                "guard %.2f ms median, screws %d",
                ticks_, frames_seen_, median_of(frame_ms_),
                frame_ms_.empty() ? 0.0 : *std::max_element(frame_ms_.begin(), frame_ms_.end()),
                median_of(guard_ms_), summary.screws_driven);
    RCLCPP_INFO(get_logger(), "wrote %s", path.c_str());
  }

  int seed_ = 0;
  std::string out_dir_;
  std::string model_path_;
  std::map<std::string, Json> knowledge_;
  Grid background_;
  bool ready_ = false;
  double fps_ = sightline::kFps;

  sightline::Calibration calib_;
  sightline::Taught taught_;
  std::vector<sightline::Plane> planes_;
  ModelPtr arm_model_;
  ModelPtr render_model_;
  std::unique_ptr<sightline::ToolKinematics> kin_;
  std::unique_ptr<sightline::KnownWorld> known_;
  std::unique_ptr<sightline::ViewGrid> grid_;
  std::unique_ptr<sightline::B4> b4_;
  std::unique_ptr<sightline::MotionLayer> motion_;
  std::unique_ptr<sightline::Perception> per_;
  Points hidden_ = Points(0, 3);

  std::map<int64_t, Grid> depths_;       // by picture time in ms, until its frame message arrives
  std::map<int64_t, Json> frames_;
  int last_frame_ = 0;
  std::optional<Json> pending_;
  int ticks_ = 0;
  int frames_seen_ = 0;
  std::vector<double> frame_ms_;
  std::vector<double> perceive_ms_;
  std::vector<double> grid_ms_;
  std::vector<double> blind_ms_;
  std::vector<double> rviz_ms_;
  std::vector<double> guard_ms_;
  std::vector<double> solve_ms_;
  std::map<std::string, int> verdicts_;
  bool done_ = false;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_calibration_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_blind_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_cell_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_background_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_frame_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_tick_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_cmd_;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pub_grid_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_ready_;
  rclcpp::Publisher<Marker>::SharedPtr pub_arm_;
  rclcpp::Publisher<Marker>::SharedPtr pub_target_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_caption_;
};

}  // namespace
}  // namespace sightline_ros

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<sightline_ros::Planner>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    while (rclcpp::ok() && !node->done()) {
      executor.spin_once(std::chrono::milliseconds(100));
    }
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("sightline_planner"), "%s", error.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
