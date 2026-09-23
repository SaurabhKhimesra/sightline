// What the two live nodes share: topic names, the JSON messages, image conversion,
// sim time stamps and the caption strip.
//
// The cell (sightline_ros/cell_node.py) and the planner (this package's planner_node)
// run in lockstep on sim time: every 10 ms the cell publishes the joint state with a
// tick, waits for the command that answers that tick, and only then steps on. Every
// 40 ms it publishes a depth picture first, and the tick after it names that picture,
// so the planner answers no tick before it has looked at the picture the cell sent.
// Nothing depends on wall time, so the run is the same run at any speed the machine
// manages.
#ifndef SIGHTLINE_ROS_LOCKSTEP_HPP
#define SIGHTLINE_ROS_LOCKSTEP_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

#include "sightline_planner/types.hpp"

namespace sightline_ros {

using Json = nlohmann::json;

// The topic contract with the cell node. The same names the Python side uses.
const std::map<std::string, std::string> &topics();
const std::string &topic(const std::string &name);

// What the cell knows once is sent once and kept for a late subscriber.
rclcpp::QoS latched();

builtin_interfaces::msg::Time stamp(double t);

std_msgs::msg::String to_json(const Json &value);
Json from_json(const std_msgs::msg::String &msg);

sensor_msgs::msg::Image depth_to_msg(const sightline::Grid &depth, double t,
                                     const std::string &frame_id = "eyes");
sightline::Grid msg_to_depth(const sensor_msgs::msg::Image &msg);

// Lines of (text, rgb) drawn on a dark strip, for an rviz Image panel.
sensor_msgs::msg::Image caption_image(
    const std::vector<std::pair<std::string, sightline::Vec3>> &lines, double t,
    int width = 960, int px = 2);

// Coloured points for rviz, straight from the point array.
sensor_msgs::msg::PointCloud2 cloud_msg(const sightline::Points &points,
                                        const sightline::Vec3 &rgb, double t,
                                        const std::string &frame_id = "world");

}  // namespace sightline_ros

#endif  // SIGHTLINE_ROS_LOCKSTEP_HPP
