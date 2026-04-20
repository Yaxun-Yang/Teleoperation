#include "allegro_hand_state_broadcaster.hpp"
#include "rclcpp/rclcpp.hpp"

AllegroHandStateBroadcaster::AllegroHandStateBroadcaster()
: Node("allegro_robot_state_broadcaster") {
    // Create publisher for joint states
    joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("allegro_hand_joint_states", 10);

    // Subscribe to Allegro hand joint commands
    joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "allegroHand_0/joint_cmd", 10, std::bind(&AllegroRobotStateBroadcaster::jointStateCallback, this, std::placeholders::_1)
    );

    // Define the joint names for the Allegro hand (16 motors)
    joint_names_ = {
      "joint_0_0", "joint_1_0", "joint_2_0", 
      "joint_3_0", "joint_4_0", "joint_5_0", 
      "joint_6_0", "joint_7_0", "joint_8_0", 
      "joint_9_0", "joint_10_0", "joint_11_0", 
      "joint_12_0", "joint_13_0", "joint_14_0", 
      "joint_15_0"
    };


    // Initialize the joint states message (16 joints/motors)
    allegro_joint_state_.name = joint_names_;
    allegro_joint_state_.position.resize(16, 0.0);  // 16 motors with initial positions (radians)
    allegro_joint_state_.velocity.resize(16, 0.0);  // Optional, depending on your control setup
    allegro_joint_state_.effort.resize(16, 0.0);    // Optional, depending on your control setup
}

void AllegroRobotStateBroadcaster::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    // Ensure that we only handle 16 joints (motors)
    if (msg->position.size() == 16) {
        // Copy the joint positions from the received message to our Allegro joint state
        allegro_joint_state_.position = msg->position;

        // Optionally, copy velocities and efforts if needed
        if (msg->velocity.size() == 16) {
            allegro_joint_state_.velocity = msg->velocity;
        }
        if (msg->effort.size() == 16) {
            allegro_joint_state_.effort = msg->effort;
        }

        // Publish the updated joint state
        joint_state_publisher_->publish(allegro_joint_state_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Received joint state message with incorrect number of positions. Expected 16, but got %zu.", msg->position.size());
    }
}
