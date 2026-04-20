#include "allegro_hand_model_broadcaster.hpp"
#include "rclcpp/rclcpp.hpp"
#include "robot_state_publisher/robot_state_publisher.hpp"
#include "urdf/model.h"

AllegroHandModelBroadcaster::AllegroHandModelBroadcaster()
: Node("allegro_robot_model_broadcaster") {

    // Define the path to the Allegro hand's URDF file
    allegro_urdf_ = "/home/basma/multipanda_ws/src/allegro_hand_controllers/urdf/allegro_hand_description_right_B.urdf";  // Replace with the actual path to your URDF file

    // Load the Allegro hand model from URDF
    urdf::Model allegro_hand_model;
    if (!allegro_hand_model.initFile(allegro_urdf_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load Allegro hand URDF from %s.", allegro_urdf_.c_str());
        return;
    }

    // Initialize robot state publisher
    state_publisher_ = std::make_shared<robot_state_publisher::RobotStatePublisher>(this);

    // Publish the robot state (transform information)
    state_publisher_->publishTransforms();
}

