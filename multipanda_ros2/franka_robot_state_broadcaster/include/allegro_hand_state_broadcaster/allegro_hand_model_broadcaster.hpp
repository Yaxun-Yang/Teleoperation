#ifndef ALLEGRO_HAND_MODEL_BROADCASTER_HPP
#define ALLEGRO_HAND_MODEL_BROADCASTER_HPP

#include "rclcpp/rclcpp.hpp"
#include "urdf/model.h"
#include "robot_state_publisher/robot_state_publisher.hpp"

allegro_urdf_ = "/home/basma/multipanda_ws/src/allegro_hand_controllers/urdf/allegro_hand_description_right_B.urdf"

class AllegroHandModelBroadcaster : public rclcpp::Node {
public:
    AllegroHandModelBroadcaster();

private:
    std::shared_ptr<robot_state_publisher::RobotStatePublisher> state_publisher_;  // Publisher for robot state
    std::string allegro_urdf_;  // Path to the Allegro hand's URDF
};

#endif  // ALLEGRO_HAND_MODEL_BROADCASTER_HPP

