#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

class ReactiveFollowGap : public rclcpp::Node {
  // Implement Reactive Follow Gap on the car

public:
  ReactiveFollowGap() : Node("reactive_node") {
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        lidarscan_topic, 10,
        std::bind(&ReactiveFollowGap::lidar_callback, this,
                  std::placeholders::_1));

    drive_pub_ =
        this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            drive_topic, 10);

    RCLCPP_INFO(this->get_logger(), "Reactive Follow Gap node started");
  }

private:
  std::string lidarscan_topic = "/scan";
  std::string drive_topic = "/drive";

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      drive_pub_;

  float prev_steer_ = 0.0f; // for steering smoothing

  // ---- Tunable parameters ----
  const int smoothing_window_ = 9;   // moving average window (samples, odd)
  const float max_range_ = 3.0f;     // clip ranges above this (m)
  const float bubble_radius_ = 0.5f; // safety bubble around closest point (m)
  const float fov_half_angle_ =
      75.0f * M_PI / 180.0f;         // only look +/- 75 deg ahead
  const float gap_threshold_ = 1.5f; // min range to be "free" (m); must be
                                     // larger than half the corridor width so
                                     // walls are NOT counted as gap
  const float best_pt_margin_ =
      0.3f; // points within this of the max count as "best"
  const float steer_alpha_ = 0.4f;  // 1.0 = no smoothing, lower = smoother
  const float max_steer_ = 0.4189f; // ~24 deg steering limit (rad)

  // Preprocess the LiDAR scan array:
  // 1. Replace NaN/Inf, clip high values
  // 2. Smooth with a moving average
  void preprocess_lidar(std::vector<float> &ranges) {
    const int n = ranges.size();

    for (int i = 0; i < n; i++) {
      if (std::isnan(ranges[i])) {
        ranges[i] = 0.0f;
      } else if (std::isinf(ranges[i]) || ranges[i] > max_range_) {
        ranges[i] = max_range_;
      }
    }

    std::vector<float> smoothed(n, 0.0f);
    const int half = smoothing_window_ / 2;
    for (int i = 0; i < n; i++) {
      float sum = 0.0f;
      int count = 0;
      for (int j = i - half; j <= i + half; j++) {
        if (j >= 0 && j < n) {
          sum += ranges[j];
          count++;
        }
      }
      smoothed[i] = sum / count;
    }
    ranges = smoothed;
  }

  // Return the start index & end index (inclusive) of the max gap in ranges,
  // searching only within [lo, hi]. A gap is a run of consecutive values
  // above gap_threshold_.
  void find_max_gap(const std::vector<float> &ranges, int lo, int hi,
                    int *indice) {
    int best_start = lo, best_end = lo, best_len = 0;
    int cur_start = -1;

    for (int i = lo; i <= hi + 1; i++) {
      bool free_pt = (i <= hi) && (ranges[i] > gap_threshold_);
      if (free_pt) {
        if (cur_start < 0)
          cur_start = i;
      } else if (cur_start >= 0) {
        int len = i - cur_start;
        if (len > best_len) {
          best_len = len;
          best_start = cur_start;
          best_end = i - 1;
        }
        cur_start = -1;
      }
    }

    indice[0] = best_start;
    indice[1] = best_end;
  }

  // Return index of best point in the gap.
  // Instead of the single furthest point (which jumps around between nearly
  // equal readings and causes oscillation), take the centroid of all points
  // that are within best_pt_margin_ of the furthest range in the gap.
  int find_best_point(const std::vector<float> &ranges, const int *indice) {
    const int start_i = indice[0];
    const int end_i = indice[1];

    float max_val = 0.0f;
    for (int i = start_i; i <= end_i; i++) {
      max_val = std::max(max_val, ranges[i]);
    }

    double sum_idx = 0.0;
    int count = 0;
    for (int i = start_i; i <= end_i; i++) {
      if (ranges[i] >= max_val - best_pt_margin_) {
        sum_idx += i;
        count++;
      }
    }
    return static_cast<int>(std::round(sum_idx / count));
  }

  void
  lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    std::vector<float> ranges(scan_msg->ranges.begin(), scan_msg->ranges.end());
    const int n = ranges.size();
    if (n == 0)
      return;

    const float angle_min = scan_msg->angle_min;
    const float angle_inc = scan_msg->angle_increment;

    preprocess_lidar(ranges);

    // Restrict to the front field of view
    int lo = static_cast<int>((-fov_half_angle_ - angle_min) / angle_inc);
    int hi = static_cast<int>((fov_half_angle_ - angle_min) / angle_inc);
    lo = std::max(lo, 0);
    hi = std::min(hi, n - 1);
    if (lo >= hi)
      return;

    // Find closest point to LiDAR (ignore invalid zero readings)
    int closest_idx = lo;
    float closest_dist = std::numeric_limits<float>::max();
    for (int i = lo; i <= hi; i++) {
      if (ranges[i] > 0.01f && ranges[i] < closest_dist) {
        closest_dist = ranges[i];
        closest_idx = i;
      }
    }

    // Eliminate all points inside 'bubble' (set them to zero)
    if (closest_dist < std::numeric_limits<float>::max()) {
      const int bubble_idx = static_cast<int>(
          std::atan2(bubble_radius_, closest_dist) / angle_inc);
      const int b_lo = std::max(closest_idx - bubble_idx, 0);
      const int b_hi = std::min(closest_idx + bubble_idx, n - 1);
      for (int i = b_lo; i <= b_hi; i++) {
        ranges[i] = 0.0f;
      }
    }

    // Find max length gap
    int gap[2];
    find_max_gap(ranges, lo, hi, gap);

    // Find the best point in the gap
    const int best_idx = find_best_point(ranges, gap);

    // Convert index to steering angle, smooth it, and clamp
    float target_steer = angle_min + best_idx * angle_inc;
    float steering_angle =
        steer_alpha_ * target_steer + (1.0f - steer_alpha_) * prev_steer_;
    steering_angle = std::clamp(steering_angle, -max_steer_, max_steer_);
    prev_steer_ = steering_angle;

    // Slow down in sharper turns
    const float abs_steer = std::fabs(steering_angle);
    float speed;
    if (abs_steer < 10.0f * M_PI / 180.0f) {
      speed = 2.5f;
    } else if (abs_steer < 20.0f * M_PI / 180.0f) {
      speed = 2.0f;
    } else {
      speed = 1.6f;
    }

    // Publish Drive message
    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.header.stamp = this->now();
    drive_msg.header.frame_id = "base_link";
    drive_msg.drive.steering_angle = steering_angle;
    drive_msg.drive.speed = speed;
    drive_pub_->publish(drive_msg);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ReactiveFollowGap>());
  rclcpp::shutdown();
  return 0;
}