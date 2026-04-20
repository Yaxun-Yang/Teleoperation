#ifndef ALLEGRO_HAND_STATE_BROADCASTER_HPP
#define ALLEGRO_HAND_STATE_BROADCASTER_HPP

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

joint_names_ = {
    "joint_0_0", "joint_1_0", "joint_2_0", 
    "joint_3_0", "joint_4_0", "joint_5_0", 
    "joint_6_0", "joint_7_0", "joint_8_0", 
    "joint_9_0", "joint_10_0", "joint_11_0", 
    "joint_12_0", "joint_13_0", "joint_14_0", 
    "joint_15_0"
};


class AllegroHandStateBroadcaster : public rclcpp::Node {
public:
    AllegroHandStateBroadcaster();
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

private:
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;
    sensor_msgs::msg::JointState allegro_joint_state_;  // Joint state for the Allegro hand
    std::vector<std::string> joint_names_;  // Names of the 16 joints (motors) for the Allegro hand
};

#endif  // ALLEGRO_HAND_STATE_BROADCASTER_HPP
